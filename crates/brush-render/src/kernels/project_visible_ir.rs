//! Compute projected splat data for visible gaussians using IR intensity
//! instead of SH coefficients. Reads `raw_ir` and outputs the splat with
//! grayscale color = [sigmoid(raw_ir), sigmoid(raw_ir), sigmoid(raw_ir)].
//! Reuses the geometry computation (cov2d, conic, projection) from PF.

use super::helpers::{
    calc_cov2d, compensate_cov2d, is_finite_f32, read_quat_unorm, read_scale, sigmoid,
    world_to_cam, write_projected_splat,
};
use super::types::{ProjectUniforms, Splat, Vec3A};
use crate::kernels::camera_model::{CameraModel, project};
use burn_cubecl::cubecl;
use burn_cubecl::cubecl::cube;
use burn_cubecl::cubecl::prelude::*;

pub const WG_SIZE: u32 = 256;

#[allow(clippy::semicolon_if_nothing_returned)]
#[cube(launch)]
pub fn project_visible_ir_kernel(
    transforms: &Tensor<f32>,
    raw_ir: &Tensor<f32>,
    raw_opacities: &Tensor<f32>,
    global_from_compact_gid: &Tensor<u32>,
    projected: &mut Tensor<f32>,
    u: ProjectUniforms,
    #[comptime] mip_splatting: bool,
    #[comptime] camera_model: CameraModel,
) {
    let compact_gid = ABSOLUTE_POS as u32;
    if compact_gid >= u.num_visible {
        terminate!();
    }

    let global_gid = global_from_compact_gid[compact_gid as usize];

    // means(3) + quats(4) + log_scales(3)
    let base = (global_gid * 10u32) as usize;
    let mean = Vec3A::new(transforms[base], transforms[base + 1], transforms[base + 2]);
    let scale = read_scale(transforms, base);
    let quat_unorm = read_quat_unorm(transforms, base);
    let quat = quat_unorm.normalize();

    let mean_c = world_to_cam(mean, u);
    let raw_cov = calc_cov2d(scale, quat, mean_c, u, camera_model);
    let (cov, filter_comp) = compensate_cov2d(raw_cov, mip_splatting);
    let opac = sigmoid(raw_opacities[global_gid as usize]) * filter_comp;
    let conic = cov.inverse();

    let (mean2d_x, mean2d_y) = project(mean_c, u.pinhole_params, camera_model);

    // IR intensity from sigmoid(raw_ir)
    let ir_val = sigmoid(raw_ir[global_gid as usize]);
    // Clamp to prevent NaN/Inf in backward
    let ir_c = clamp(select(is_finite_f32(ir_val), ir_val, 0.5f32), 0.0f32, 1.0f32);

    write_projected_splat(
        projected,
        compact_gid,
        Splat {
            xy_x: mean2d_x,
            xy_y: mean2d_y,
            conic_x: conic.c00,
            conic_y: conic.c01,
            conic_z: conic.c11,
            color_a: opac,
            color_r: ir_c,
            color_g: ir_c,
            color_b: ir_c,
        },
    );
}
