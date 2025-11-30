use bevy_app::{App, Plugin};
use bevy_asset::{Asset, AssetApp, Handle};
use bevy_color::{Color, LinearRgba};
use bevy_ecs::{component::Component, reflect::ReflectComponent};
use bevy_image::Image;
use bevy_math::Affine2;
use bevy_reflect::{std_traits::ReflectDefault, Reflect, TypePath};
use wgpu_types::Face;

use crate::alpha::AlphaMode;


/// An enum to define which UV attribute to use for a texture.
///
/// It is used for every texture in the [`StandardMaterial`].
/// It only supports two UV attributes, [`bevy_mesh::Mesh::ATTRIBUTE_UV_0`] and
/// [`bevy_mesh::Mesh::ATTRIBUTE_UV_1`].
/// The default is [`UvChannel::Uv0`].
#[derive(Reflect, Default, Debug, Clone, PartialEq, Eq)]
#[reflect(Default, Debug, Clone, PartialEq)]
pub enum UvChannel {
    #[default]
    Uv0,
    Uv1,
}


#[derive(Component, Clone, Debug, Reflect)]
#[reflect(Component, Default, Clone, PartialEq)]
pub struct MarkerMeshMaterial3d(pub Handle<ShortStandardMaterial>);

impl Default for MarkerMeshMaterial3d {
    fn default() -> Self {
        Self(Handle::default())
    }
}

impl PartialEq for MarkerMeshMaterial3d {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for MarkerMeshMaterial3d {}

// impl From<MarkerMeshMaterial3d> for AssetId<ShortStandardMaterial> {
//     fn from(material: MarkerMeshMaterial3d) -> Self {
//         material.id()
//     }
// }

// impl From<&MarkerMeshMaterial3d> for AssetId<ShortStandardMaterial> {
//     fn from(material: &MarkerMeshMaterial3d) -> Self {
//         material.id()
//     }
// }

// impl AsAssetId for MarkerMeshMaterial3d {
//     type Asset = ShortStandardMaterial;

//     fn as_asset_id(&self) -> AssetId<Self::Asset> {
//         self.id()
//     }
// }



// Data to build a bevy_pbr::StandardMaterial
#[derive(Asset, Debug, TypePath)]
pub struct ShortStandardMaterial {
    pub base_color: Color,
    pub base_color_channel: UvChannel,
    pub base_color_texture: Option<Handle<Image>>,
    pub emissive: LinearRgba,
    pub emissive_channel: UvChannel,
    pub emissive_texture: Option<Handle<Image>>,
    pub perceptual_roughness: f32,
    pub metallic: f32,
    pub metallic_roughness_channel: UvChannel,
    pub metallic_roughness_texture: Option<Handle<Image>>,
    pub reflectance: f32,
    pub specular_tint: Color,
    pub specular_transmission: f32,
    #[cfg(feature = "pbr_transmission_textures")]
    pub specular_transmission_channel: UvChannel,
    #[cfg(feature = "pbr_transmission_textures")]
    pub specular_transmission_texture: Option<Handle<Image>>,
    pub thickness: f32,
    #[cfg(feature = "pbr_transmission_textures")]
    pub thickness_channel: UvChannel,
    #[cfg(feature = "pbr_transmission_textures")]
    pub thickness_texture: Option<Handle<Image>>,
    pub ior: f32,
    pub attenuation_distance: f32,
    pub attenuation_color: Color,
    pub normal_map_channel: UvChannel,
    pub normal_map_texture: Option<Handle<Image>>,
    pub occlusion_channel: UvChannel,
    pub occlusion_texture: Option<Handle<Image>>,
    pub clearcoat: f32,
    pub clearcoat_perceptual_roughness: f32,
    pub anisotropy_strength: f32,
    pub anisotropy_rotation: f32,
    pub double_sided: bool,
    pub cull_mode: Option<Face>,
    pub unlit: bool,
    pub alpha_mode: AlphaMode,
    pub uv_transform: Affine2,
}

impl Default for ShortStandardMaterial {
    fn default() -> Self {
        ShortStandardMaterial {
            // White because it gets multiplied with texture values if someone uses
            // a texture.
            base_color: Color::WHITE,
            base_color_channel: UvChannel::Uv0,
            base_color_texture: None,
            emissive: LinearRgba::BLACK,
            emissive_channel: UvChannel::Uv0,
            emissive_texture: None,
            // Matches Blender's default roughness.
            perceptual_roughness: 0.5,
            // Metallic should generally be set to 0.0 or 1.0.
            metallic: 0.0,
            metallic_roughness_channel: UvChannel::Uv0,
            metallic_roughness_texture: None,
            // Minimum real-world reflectance is 2%, most materials between 2-5%
            // Expressed in a linear scale and equivalent to 4% reflectance see
            // <https://google.github.io/filament/Material%20Properties.pdf>
            reflectance: 0.5,
            specular_transmission: 0.0,
            #[cfg(feature = "pbr_transmission_textures")]
            specular_transmission_channel: UvChannel::Uv0,
            #[cfg(feature = "pbr_transmission_textures")]
            specular_transmission_texture: None,
            thickness: 0.0,
            #[cfg(feature = "pbr_transmission_textures")]
            thickness_channel: UvChannel::Uv0,
            #[cfg(feature = "pbr_transmission_textures")]
            thickness_texture: None,
            ior: 1.5,
            attenuation_color: Color::WHITE,
            attenuation_distance: f32::INFINITY,
            occlusion_channel: UvChannel::Uv0,
            occlusion_texture: None,
            normal_map_channel: UvChannel::Uv0,
            normal_map_texture: None,
            #[cfg(feature = "pbr_specular_textures")]
            specular_channel: UvChannel::Uv0,
            #[cfg(feature = "pbr_specular_textures")]
            specular_texture: None,
            specular_tint: Color::WHITE,
            #[cfg(feature = "pbr_specular_textures")]
            specular_tint_channel: UvChannel::Uv0,
            #[cfg(feature = "pbr_specular_textures")]
            specular_tint_texture: None,
            clearcoat: 0.0,
            clearcoat_perceptual_roughness: 0.5,
            #[cfg(feature = "pbr_multi_layer_material_textures")]
            clearcoat_channel: UvChannel::Uv0,
            #[cfg(feature = "pbr_multi_layer_material_textures")]
            clearcoat_texture: None,
            #[cfg(feature = "pbr_multi_layer_material_textures")]
            clearcoat_roughness_channel: UvChannel::Uv0,
            #[cfg(feature = "pbr_multi_layer_material_textures")]
            clearcoat_roughness_texture: None,
            #[cfg(feature = "pbr_multi_layer_material_textures")]
            clearcoat_normal_channel: UvChannel::Uv0,
            #[cfg(feature = "pbr_multi_layer_material_textures")]
            clearcoat_normal_texture: None,
            anisotropy_strength: 0.0,
            anisotropy_rotation: 0.0,
            #[cfg(feature = "pbr_anisotropy_texture")]
            anisotropy_channel: UvChannel::Uv0,
            #[cfg(feature = "pbr_anisotropy_texture")]
            anisotropy_texture: None,
            double_sided: false,
            cull_mode: Some(Face::Back),
            unlit: false,
            alpha_mode: AlphaMode::Opaque,
            uv_transform: Affine2::IDENTITY,
        }
    }
}



#[derive(Default)]
pub struct ShortMaterialPlugin;

impl Plugin for ShortMaterialPlugin {
    fn build(&self, app: &mut App) {
        app.init_asset::<ShortStandardMaterial>();
    }
}
