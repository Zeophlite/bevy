//! Create an application without winit (runs single time, no event loop).

use bevy::{prelude::*, winit::WinitPlugin};

fn main() {
    Bevy::new()
        .add_plugins(DefaultPlugins.build().disable::<WinitPlugin>())
        .add_systems(Update, setup_system)
        .run();
}

fn setup_system(mut commands: Commands) {
    commands.spawn(Camera3d::default());
}
