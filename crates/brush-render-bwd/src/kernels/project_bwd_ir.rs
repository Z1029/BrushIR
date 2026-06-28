//! Backward pass for IR projection.
//!
//! Reads the color-channel gradients from `v_combined` (slots 6,7,8 = r,g,b),
//! sums them (since IR uses grayscale = same value for r,g,b), multiplies by
//! the sigmoid derivative of `raw_ir`, and scatters to the dense output.
//!
//! This is the **only** backward kernel needed for IR training — transforms,
//! rotations, scales, SH and opacity are frozen in Stage 2.

use brush_cube::{is_finite_f32, sigmoid};
use brush_render::kernels::helpers::PROJECTED_LANES;
use brush_render::kernels::types::ProjectUniforms;
use burn_cubecl::cubecl;
use burn_cubecl::cubecl::cube;
use burn_cubecl::cubecl::prelude::*;

pub const WG_SIZE: u32 = 256;

#[cube(launch)]
pub fn project_bwd_ir_kernel(
    raw_ir: &Tensor<f32>,
    global_from_compact_gid: &Tensor<u32>,
    v_combined: &Tensor<f32>,
    v_raw_ir: &mut Tensor<f32>,
    u: ProjectUniforms,
) {
    let compact_gid = ABSOLUTE_POS as u32;
    if compact_gid >= u.num_visible {
        terminate!();
    }

    let global_gid = global_from_compact_gid[compact_gid as usize];
    let base = (compact_gid * PROJECTED_LANES) as usize;

    // Gradients for the projected color channels (r=slot6, g=slot7, b=slot8).
    // IR puts the same value in all three, so dL/d(ir_val) = sum of the three.
    let color_r_grad = v_combined[base + 6usize];
    let color_g_grad = v_combined[base + 7usize];
    let color_b_grad = v_combined[base + 8usize];
    let dir_grad = color_r_grad + color_g_grad + color_b_grad;

    // sigmoid derivative: sigmoid(x) * (1 - sigmoid(x))
    let ir = sigmoid(raw_ir[global_gid as usize]);
    let ir_deriv = ir * (1.0f32 - ir);

    // Chain rule: dL/d(raw_ir) = dL/d(ir_val) * d(ir_val)/d(raw_ir)
    let grad = dir_grad * ir_deriv;

    // Replace NaN/Inf with zero and write directly.
    // Each compact_gid maps to a unique global splat, so no atomics needed.
    v_raw_ir[global_gid as usize] = select(is_finite_f32(grad), grad, 0.0f32);
}
