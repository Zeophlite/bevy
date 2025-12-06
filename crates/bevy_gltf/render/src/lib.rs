#![cfg_attr(docsrs, feature(doc_cfg))]
#![forbid(unsafe_code)]
#![doc(
    html_logo_url = "https://bevy.org/assets/icon.png",
    html_favicon_url = "https://bevy.org/assets/icon.png"
)]

//! Plugin for rendering GLTF's using PBR

use bevy_app::{App, Plugin, PreUpdate};


pub mod render;

pub use render::*;

#[derive(Default)]
pub struct GltfRenderPlugin;

impl Plugin for GltfRenderPlugin {
    fn build(&self, app: &mut App) {
        // TODO: set this schedule to just after the Marker gets added
        app.add_systems(PreUpdate, swap_marker_mesh_material_3d);
    }
}

