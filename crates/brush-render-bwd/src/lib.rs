pub mod burn_glue;
mod kernels;
mod render_bwd;
mod render_ir_bwd;

pub use burn_glue::{
    RasterizeGrads, SplatBwdOps, SplatGrads, SplatOutputDiff, render_splats,
    render_splats_with_pass,
};
pub use render_ir_bwd::{IrOutputDiff, SplatIrBwdOps, render_ir_splats};
