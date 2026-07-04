//! An empty application with default plugins.

use bevy::prelude::*;

fn main() {
    Bevy::new().add_plugins(DefaultPlugins).run();
}
