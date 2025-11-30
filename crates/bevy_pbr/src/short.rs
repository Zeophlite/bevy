use bevy_app::{App, Plugin, PreUpdate};
use bevy_asset::Assets;
use bevy_ecs::{
    entity::Entity,
    system::{Commands, Query, Res, ResMut},
};
use bevy_material::pbr_material::{MarkerMeshMaterial3d, ShortStandardMaterial};

use crate::{MeshMaterial3d, StandardMaterial};

pub struct ShortPlugin;

impl Plugin for ShortPlugin {
    fn build(&self, app: &mut App) {
        // TODO: set this schedule to just after the Marker gets added
        app.add_systems(PreUpdate, swap_marker_mesh_material_3d);
    }
}

fn material_from_short_material(material: &ShortStandardMaterial) -> StandardMaterial {
    StandardMaterial {
        base_color: material.base_color,
        base_color_channel: material.base_color_channel.clone(),
        base_color_texture: material.base_color_texture.clone(),
        emissive: material.emissive,
        emissive_channel: material.emissive_channel.clone(),
        emissive_texture: material.emissive_texture.clone(),
        perceptual_roughness: material.perceptual_roughness,
        metallic: material.metallic,
        metallic_roughness_channel: material.metallic_roughness_channel.clone(),
        metallic_roughness_texture: material.metallic_roughness_texture.clone(),
        reflectance: material.reflectance,
        specular_tint: material.specular_tint,
        specular_transmission: material.specular_transmission,
        #[cfg(feature = "pbr_transmission_textures")]
        specular_transmission_channel: material.specular_transmission_channel.clone(),
        #[cfg(feature = "pbr_transmission_textures")]
        specular_transmission_texture: material.specular_transmission_texture.clone(),
        thickness: material.thickness,
        #[cfg(feature = "pbr_transmission_textures")]
        thickness_channel: material.thickness_channel.clone(),
        #[cfg(feature = "pbr_transmission_textures")]
        thickness_texture: material.thickness_texture.clone(),
        ior: material.ior,
        attenuation_distance: material.attenuation_distance,
        attenuation_color: material.attenuation_color,
        normal_map_channel: material.normal_map_channel.clone(),
        normal_map_texture: material.normal_map_texture.clone(),
        occlusion_channel: material.occlusion_channel.clone(),
        occlusion_texture: material.occlusion_texture.clone(),
        clearcoat: material.clearcoat,
        clearcoat_perceptual_roughness: material.clearcoat_perceptual_roughness,
        anisotropy_strength: material.anisotropy_strength,
        anisotropy_rotation: material.anisotropy_rotation,
        double_sided: material.double_sided,
        cull_mode: material.cull_mode,
        unlit: material.unlit,
        alpha_mode: material.alpha_mode,
        uv_transform: material.uv_transform,
        ..Default::default()
    }
}

fn swap_marker_mesh_material_3d(
    mut commands: Commands,
    query: Query<(Entity, &MarkerMeshMaterial3d)>,
    short_materials: Res<Assets<ShortStandardMaterial>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    for (entity, marker) in &query {
        let content = &marker.0;

        let short_material = short_materials.get(content).unwrap();

        let material = material_from_short_material(short_material);

        let content2 = materials.add(material);

        let mesh_material_3d = MeshMaterial3d::<StandardMaterial>(content2);

        commands
            .entity(entity)
            .remove::<MarkerMeshMaterial3d>()
            .insert(mesh_material_3d);
    }
}
