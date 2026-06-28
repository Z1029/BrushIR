use brush_render::AlphaMode;
use clap::Args;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Args, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct ModelConfig {
    /// SH degree of splats.
    #[arg(
        long,
        help_heading = "Model Options",
        default_value = "3",
        value_parser = clap::value_parser!(u32).range(0..=4)
    )]
    pub sh_degree: u32,
}

#[derive(Clone, Debug, Args, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct LoadDataseConfig {
    /// Max nr. of frames of dataset to load
    #[arg(long, help_heading = "Dataset Options")]
    pub max_frames: Option<usize>,
    /// Max resolution of images to load.
    #[arg(long, help_heading = "Dataset Options", default_value = "1920")]
    pub max_resolution: u32,
    /// Create an eval dataset by selecting every nth image
    #[arg(long, help_heading = "Dataset Options")]
    pub eval_split_every: Option<usize>,
    /// Load only every nth frame
    #[arg(long, help_heading = "Dataset Options")]
    pub subsample_frames: Option<u32>,
    /// Load only every nth point from the initial sfm data
    #[arg(long, help_heading = "Dataset Options")]
    pub subsample_points: Option<u32>,
    /// Whether to interpret an alpha channel (or masks) as transparency or masking.
    #[arg(long, help_heading = "Dataset Options")]
    pub alpha_mode: Option<AlphaMode>,
    /// IR subdirectory name (e.g. "ir"). IR images loaded when set.
    #[arg(long, help_heading = "IR Options", default_value = None)]
    pub ir_subdir: Option<String>,
    /// Translation offset from RGB camera to IR camera (meters). [x, y, z].
    #[arg(long, help_heading = "IR Options", default_value = "0.0", allow_hyphen_values = true, num_args = 3)]
    pub ir_translation_offset: Vec<f32>,
    /// Rotation offset (quaternion) from RGB camera to IR camera. [w, x, y, z].
    #[arg(long, help_heading = "IR Options", default_value = "1.0", num_args = 4)]
    pub ir_rotation_offset: Vec<f32>,
}
