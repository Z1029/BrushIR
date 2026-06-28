#![allow(clippy::match_wildcard_for_single_variants)]

use brush_cube::{MainBackend, MainBackendBase, calc_cube_count_1d};
use brush_render::burn_glue::{AutodiffMain, unwrap_ad_wgpu_float, wrap_ad_wgpu_float};
use brush_render::camera::Camera;
use brush_render::gaussian_splats::{SplatRenderMode, Splats};
use brush_render::SplatOps;
use brush_render::shaders::helpers::ProjectUniforms;
use burn::{
    backend::{
        Backend, TensorMetadata,
        autodiff::{
            checkpoint::{base::Checkpointer, strategy::NoCheckpointing},
            grads::Gradients,
            ops::{Backward, Ops, OpsKind},
        },
        ops::FloatTensorOps,
        tensor::{FloatTensor, IntTensor},
    },
    tensor::Tensor,
};
use crate::burn_glue::SplatBwdOps;
use burn_cubecl::cubecl::CubeDim;
use burn_cubecl::kernel::into_contiguous;
use burn_wgpu::WgpuRuntime;
use glam::Vec3;

pub struct IrOutputDiff {
    /// Rendered IR image, on the autodiff graph.
    pub img: Tensor<3>,
    pub num_visible: u32,
}

/// Backward pass trait for IR rendering.
pub trait SplatIrBwdOps: SplatOps {
    /// Backward pass for IR projection.
    /// Takes the sparse `v_combined` from [`SplatBwdOps::rasterize_bwd`] and
    /// computes the gradient w.r.t. `raw_ir` only.
    #[allow(clippy::too_many_arguments)]
    fn project_bwd_ir(
        raw_ir: FloatTensor<Self>,
        global_from_compact_gid: IntTensor<Self>,
        project_uniforms: ProjectUniforms,
        v_combined: FloatTensor<Self>,
    ) -> FloatTensor<Self>;
}

/// State saved during the IR forward pass for backward.
#[derive(Debug, Clone)]
struct IrBackwardState<B: Backend> {
    raw_ir: FloatTensor<B>,
    projected_splats: FloatTensor<B>,
    project_uniforms: ProjectUniforms,
    global_from_compact_gid: IntTensor<B>,
    out_img: FloatTensor<B>,
    compact_gid_from_isect: IntTensor<B>,
    tile_offsets: IntTensor<B>,
    background: Vec3,
    img_size: glam::UVec2,
}

#[derive(Debug)]
struct RenderIrBackwards;

const NUM_IR_BWD_ARGS: usize = 1; // only raw_ir

impl<B: Backend + SplatIrBwdOps + SplatBwdOps> Backward<B, NUM_IR_BWD_ARGS> for RenderIrBackwards {
    type State = IrBackwardState<B>;

    fn backward(
        self,
        ops: Ops<Self::State, NUM_IR_BWD_ARGS>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        let _span = tracing::trace_span!("render_ir backwards").entered();

        let state = ops.state;
        let v_output = grads.consume::<B>(&ops.node);

        let [raw_ir_parent] = ops.parents;

        // Reuse the existing rasterize_bwd to get v_combined
        let rasterize_grads = B::rasterize_bwd(
            state.out_img,
            state.projected_splats,
            state.compact_gid_from_isect,
            state.tile_offsets,
            state.background,
            state.img_size,
            v_output,
            false, // smooth_cutoff = false for IR
        );

        // Only compute raw_ir gradient — transforms/SH/opacity are frozen
        let v_raw_ir = B::project_bwd_ir(
            state.raw_ir,
            state.global_from_compact_gid,
            state.project_uniforms,
            rasterize_grads.v_combined,
        );

        if let Some(node) = raw_ir_parent {
            grads.register::<B>(node.id, v_raw_ir);
        }
    }
}

/// Render IR on a differentiable device.
///
/// The geometry pipeline (transforms, opacity) is **not** differentiated —
/// only `raw_ir` gets gradients. This is used in Stage 2 training after
/// RGB geometry has converged.
pub async fn render_ir_splats(
    splats: Splats,
    camera: &Camera,
    img_size: glam::UVec2,
    background: Vec3,
) -> IrOutputDiff {
    let device = splats.device();
    assert!(
        device.is_autodiff(),
        "render_ir_splats requires an autodiff-enabled device"
    );

    let raw_ir_val = splats.raw_ir.val();
    let raw_ir_ad = unwrap_ad_wgpu_float(raw_ir_val);

    let prep_nodes = RenderIrBackwards
        .prepare::<NoCheckpointing>([raw_ir_ad.node.clone()])
        .compute_bound()
        .stateful();

    let render_mode = if splats.render_mip {
        SplatRenderMode::Mip
    } else {
        SplatRenderMode::Default
    };

    let transforms_val = splats.transforms.val();
    let raw_opac_val = splats.raw_opacities.val();

    let transforms_inner = unwrap_ad_wgpu_float(transforms_val).primitive;
    let raw_ir_inner = raw_ir_ad.primitive.clone();
    let raw_opac_inner = unwrap_ad_wgpu_float(raw_opac_val).primitive;

    let output = <MainBackend as SplatOps>::render_ir(
        camera,
        img_size,
        transforms_inner.clone(),
        raw_ir_inner.clone(),
        raw_opac_inner.clone(),
        render_mode,
        background,
    )
    .await;

    let img_ad: FloatTensor<AutodiffMain> = match prep_nodes {
        OpsKind::Tracked(prep) => {
            let state = IrBackwardState {
                raw_ir: raw_ir_inner,
                projected_splats: output.projected_splats,
                project_uniforms: output.project_uniforms,
                global_from_compact_gid: output.global_from_compact_gid,
                out_img: output.out_img.clone(),
                compact_gid_from_isect: output.compact_gid_from_isect,
                tile_offsets: output.aux.tile_offsets.clone(),
                background,
                img_size,
            };
            prep.finish(state, output.out_img)
        }
        OpsKind::UnTracked(prep) => prep.finish(output.out_img),
    };

    IrOutputDiff {
        img: wrap_ad_wgpu_float(img_ad),
        num_visible: output.aux.num_visible,
    }
}

impl SplatIrBwdOps for MainBackendBase {
    fn project_bwd_ir(
        raw_ir: FloatTensor<Self>,
        global_from_compact_gid: IntTensor<Self>,
        project_uniforms: ProjectUniforms,
        v_combined: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        let _span = tracing::trace_span!("project_bwd_ir").entered();

        let raw_ir = into_contiguous(raw_ir);
        let device = raw_ir.device.clone();
        let num_points = raw_ir.shape()[0];
        let client = raw_ir.client.clone();

        let v_raw_ir = Self::float_zeros([num_points].into(), &device, burn::tensor::FloatDType::F32);

        let num_visible = project_uniforms.num_visible;
        let uniforms = project_uniforms.to_launch_object();

        tracing::trace_span!("ProjectBwdIr").in_scope(|| {
            crate::kernels::project_bwd_ir::project_bwd_ir_kernel::launch::<WgpuRuntime>(
                &client,
                calc_cube_count_1d(num_visible, crate::kernels::project_bwd_ir::WG_SIZE),
                CubeDim::new_1d(crate::kernels::project_bwd_ir::WG_SIZE),
                raw_ir.into_tensor_arg(),
                global_from_compact_gid.into_tensor_arg(),
                v_combined.into_tensor_arg(),
                v_raw_ir.clone().into_tensor_arg(),
                uniforms,
            );
        });

        v_raw_ir
    }
}
