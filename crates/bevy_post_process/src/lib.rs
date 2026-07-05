#![expect(missing_docs, reason = "Not all docs are written yet, see #3492.")]
#![forbid(unsafe_code)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![doc(
    html_logo_url = "https://bevy.org/assets/icon.png",
    html_favicon_url = "https://bevy.org/assets/icon.png"
)]

pub mod auto_exposure;
pub mod bloom;
pub mod dof;
pub mod effect_stack;
pub mod motion_blur;
pub mod msaa_writeback;

use crate::{
    bloom::BloomPlugin, dof::DepthOfFieldPlugin, effect_stack::EffectStackPlugin,
    motion_blur::MotionBlurPlugin, msaa_writeback::MsaaWritebackPlugin,
};
use bevy_app::{Bevy, Plugin};
use bevy_shader::load_shader_library;

/// Adds bloom, motion blur, depth of field, and chromatic aberration support.
#[derive(Default)]
pub struct PostProcessPlugin;

impl Plugin for PostProcessPlugin {
    fn build(&self, bevy: &mut Bevy) {
        let app = bevy.main_mut();
        load_shader_library!(app, "gaussian_blur.wgsl");

        app.add_plugins((
            MsaaWritebackPlugin,
            BloomPlugin,
            MotionBlurPlugin,
            DepthOfFieldPlugin,
            EffectStackPlugin,
        ));
    }
}
