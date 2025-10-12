//! Lightmaps, baked lighting textures that can be applied at runtime to provide
//! diffuse global illumination.
//!
//! Bevy doesn't currently have any way to actually bake lightmaps, but they can
//! be baked in an external tool like [Blender](http://blender.org), for example
//! with an addon like [The Lightmapper]. The tools in the [`bevy-baked-gi`]
//! project support other lightmap baking methods.
//!
//! When a [`Lightmap`] component is added to an entity with a [`Mesh3d`] and a
//! [`MeshMaterial3d<StandardMaterial>`], Bevy applies the lightmap when rendering. The brightness
//! of the lightmap may be controlled with the `lightmap_exposure` field on
//! [`StandardMaterial`].
//!
//! During the rendering extraction phase, we extract all lightmaps into the
//! [`RenderLightmaps`] table, which lives in the render world. Mesh bindgroup
//! and mesh uniform creation consults this table to determine which lightmap to
//! supply to the shader. Essentially, the lightmap is a special type of texture
//! that is part of the mesh instance rather than part of the material (because
//! multiple meshes can share the same material, whereas sharing lightmaps is
//! nonsensical).
//!
//! Note that multiple meshes can't be drawn in a single drawcall if they use
//! different lightmap textures, unless bindless textures are in use. If you
//! want to instance a lightmapped mesh, and your platform doesn't support
//! bindless textures, combine the lightmap textures into a single atlas, and
//! set the `uv_rect` field on [`Lightmap`] appropriately.
//!
//! [The Lightmapper]: https://github.com/Naxela/The_Lightmapper
//! [`Mesh3d`]: bevy_mesh::Mesh3d
//! [`MeshMaterial3d<StandardMaterial>`]: crate::StandardMaterial
//! [`StandardMaterial`]: crate::StandardMaterial
//! [`bevy-baked-gi`]: https://github.com/pcwalton/bevy-baked-gi

use bevy_app::{App, Plugin};
use bevy_asset::{AssetId, Handle};
use bevy_camera::visibility::ViewVisibility;
use bevy_derive::{Deref, DerefMut};
use bevy_ecs::{
    component::Component,
    entity::Entity,
    lifecycle::RemovedComponents,
    query::{Changed, Or},
    reflect::ReflectComponent,
    resource::Resource,
    schedule::IntoScheduleConfigs,
    system::{Commands, Query, Res, ResMut},
};
use bevy_image::Image;
use bevy_math::{uvec2, vec4, Rect, UVec2};
use bevy_platform::collections::HashSet;
use bevy_reflect::{std_traits::ReflectDefault, Reflect};
use bevy_render::{
    mesh::lightmap::{LightmapSlabIndex, LightmapSlotIndex, RenderLightmap, RenderLightmapsU}, render_asset::RenderAssets, render_resource::{Sampler, TextureView, WgpuSampler, WgpuTextureView}, renderer::RenderAdapter, sync_world::MainEntity, texture::{FallbackImage, GpuImage}, Extract, ExtractSchedule, RenderApp, RenderStartup
};
use bevy_render::{renderer::RenderDevice, sync_world::MainEntityHashMap};
use bevy_shader::load_shader_library;
use bevy_utils::default;
use fixedbitset::FixedBitSet;
use nonmax::{NonMaxU16, NonMaxU32};
use tracing::error;

use crate::{binding_arrays_are_usable, MeshExtractionSystems};

/// The number of lightmaps that we store in a single slab, if bindless textures
/// are in use.
///
/// If bindless textures aren't in use, then only a single lightmap can be bound
/// at a time.
pub const LIGHTMAPS_PER_SLAB: usize = 4;

/// A plugin that provides an implementation of lightmaps.
pub struct LightmapPlugin;

/// A component that applies baked indirect diffuse global illumination from a
/// lightmap.
///
/// When assigned to an entity that contains a [`Mesh3d`](bevy_mesh::Mesh3d) and a
/// [`MeshMaterial3d<StandardMaterial>`](crate::StandardMaterial), if the mesh
/// has a second UV layer ([`ATTRIBUTE_UV_1`](bevy_mesh::Mesh::ATTRIBUTE_UV_1)),
/// then the lightmap will render using those UVs.
#[derive(Component, Clone, Reflect)]
#[reflect(Component, Default, Clone)]
pub struct Lightmap {
    /// The lightmap texture.
    pub image: Handle<Image>,

    /// The rectangle within the lightmap texture that the UVs are relative to.
    ///
    /// The top left coordinate is the `min` part of the rect, and the bottom
    /// right coordinate is the `max` part of the rect. The rect ranges from (0,
    /// 0) to (1, 1).
    ///
    /// This field allows lightmaps for a variety of meshes to be packed into a
    /// single atlas.
    pub uv_rect: Rect,

    /// Whether bicubic sampling should be used for sampling this lightmap.
    ///
    /// Bicubic sampling is higher quality, but slower, and may lead to light leaks.
    ///
    /// If true, the lightmap texture's sampler must be set to [`bevy_image::ImageSampler::linear`].
    pub bicubic_sampling: bool,
}

/// Corresponds to `RenderLightmapsU`
#[derive(Resource)]
pub struct RenderLightmapsL {
    /// The slabs (binding arrays) containing the lightmaps.
    pub slabs: Vec<LightmapSlab>,
}

/// A binding array that contains lightmaps.
///
/// This will have a single binding if bindless lightmaps aren't in use.
pub struct LightmapSlab {
    /// The GPU images in this slab.
    lightmaps: Vec<AllocatedLightmap>,
    free_slots_bitmask: u32,
}

struct AllocatedLightmap {
    gpu_image: GpuImage,
}

impl Plugin for LightmapPlugin {
    fn build(&self, app: &mut App) {
        load_shader_library!(app, "lightmap.wgsl");

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .add_systems(RenderStartup, init_render_lightmaps)
            .add_systems(
                ExtractSchedule,
                extract_lightmaps.after(MeshExtractionSystems),
            );
    }
}

/// Extracts all lightmaps from the scene and populates the [`RenderLightmaps`]
/// resource.
fn extract_lightmaps(
    render_lightmaps_u: ResMut<RenderLightmapsU>,
    mut render_lightmaps_l: ResMut<RenderLightmapsL>,
    changed_lightmaps_query: Extract<
        Query<
            (Entity, &ViewVisibility, &Lightmap),
            Or<(Changed<ViewVisibility>, Changed<Lightmap>)>,
        >,
    >,
    mut removed_lightmaps_query: Extract<RemovedComponents<Lightmap>>,
    images: Res<RenderAssets<GpuImage>>,
    fallback_images: Res<FallbackImage>,
) {
    let render_lightmaps_u2 = render_lightmaps_u.into_inner();
    let render_lightmaps_l2 = render_lightmaps_l.into_inner();

    // Loop over each entity.
    for (entity, view_visibility, lightmap) in changed_lightmaps_query.iter() {
        if render_lightmaps_u2
            .render_lightmaps
            .contains_key(&MainEntity::from(entity))
        {
            continue;
        }

        // Only process visible entities.
        if !view_visibility.get() {
            continue;
        }

        let (slab_index, slot_index) =
            render_lightmaps_l2.allocate(render_lightmaps_u2, &fallback_images, lightmap.image.id());
        render_lightmaps_u2.render_lightmaps.insert(
            entity.into(),
            RenderLightmap::new(
                lightmap.uv_rect,
                slab_index,
                slot_index,
                lightmap.bicubic_sampling,
            ),
        );

        render_lightmaps_u2
            .pending_lightmaps
            .insert((slab_index, slot_index));
    }

    for entity in removed_lightmaps_query.read() {
        if changed_lightmaps_query.contains(entity) {
            continue;
        }

        let Some(RenderLightmap {
            slab_index,
            slot_index,
            ..
        }) = render_lightmaps_u2
            .render_lightmaps
            .remove(&MainEntity::from(entity))
        else {
            continue;
        };

        // render_lightmaps_l.remove(render_lightmaps_u2, &fallback_images, slab_index, slot_index);
        render_lightmaps_u2
            .pending_lightmaps
            .remove(&(slab_index, slot_index));
    }

    render_lightmaps_u2
        .pending_lightmaps
        .retain(|&(slab_index, slot_index)| {
            return false;
            // let asset_id = render_lightmaps_u2.slabs[usize::from(slab_index)].lightmaps
            //     [usize::from(slot_index)]
            // .asset_id
            // else {
            //     error!(
            //         "Allocated lightmap should have been removed from `pending_lightmaps` by now"
            //     );
            //     return false;
            // };

            // let Some(gpu_image) = images.get(asset_id) else {
            //     return true;
            // };
            // render_lightmaps_l.slabs[usize::from(slab_index)].insert(slot_index, gpu_image.clone());
            // false
        });
}

impl Default for Lightmap {
    fn default() -> Self {
        Self {
            image: Default::default(),
            uv_rect: Rect::new(0.0, 0.0, 1.0, 1.0),
            bicubic_sampling: false,
        }
    }
}

pub fn init_render_lightmaps(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_adapter: Res<RenderAdapter>,
) {
    let bindless_supported = binding_arrays_are_usable(&render_device, &render_adapter);

    commands.insert_resource(RenderLightmapsU {
        render_lightmaps: default(),
        slabs: vec![],
        free_slabs: FixedBitSet::new(),
        pending_lightmaps: default(),
        bindless_supported,
    });
}

impl LightmapSlab {
    fn new(fallback_images: &FallbackImage, bindless_supported: bool) -> LightmapSlab {
        let count = if bindless_supported {
            LIGHTMAPS_PER_SLAB
        } else {
            1
        };

        LightmapSlab {
            lightmaps: (0..count)
                .map(|_| AllocatedLightmap {
                    gpu_image: fallback_images.d2.clone(),
                })
                .collect(),
            free_slots_bitmask: (1 << count) - 1,
        }
    }

    fn is_full(&self) -> bool {
        self.free_slots_bitmask == 0
    }

    fn allocate(&mut self, image_id: AssetId<Image>) -> LightmapSlotIndex {
        let index = LightmapSlotIndex::from(self.free_slots_bitmask.trailing_zeros());
        self.free_slots_bitmask &= !(1 << u32::from(index));
        // self.lightmaps[usize::from(index)].asset_id = Some(image_id);
        index
    }

    fn insert(&mut self, index: LightmapSlotIndex, gpu_image: GpuImage) {
        self.lightmaps[usize::from(index)] = AllocatedLightmap {
            gpu_image,
        }
    }

    fn remove(&mut self, fallback_images: &FallbackImage, index: LightmapSlotIndex) {
        self.lightmaps[usize::from(index)] = AllocatedLightmap {
            gpu_image: fallback_images.d2.clone(),
        };
        self.free_slots_bitmask |= 1 << u32::from(index);
    }

    /// Returns the texture views and samplers for the lightmaps in this slab,
    /// ready to be placed into a bind group.
    ///
    /// This is used when constructing bind groups in bindless mode. Before
    /// returning, this function pads out the arrays with fallback images in
    /// order to fulfill requirements of platforms that require full binding
    /// arrays (e.g. DX12).
    pub(crate) fn build_binding_arrays(&self) -> (Vec<&WgpuTextureView>, Vec<&WgpuSampler>) {
        (
            self.lightmaps
                .iter()
                .map(|allocated_lightmap| &*allocated_lightmap.gpu_image.texture_view)
                .collect(),
            self.lightmaps
                .iter()
                .map(|allocated_lightmap| &*allocated_lightmap.gpu_image.sampler)
                .collect(),
        )
    }

    /// Returns the texture view and sampler corresponding to the first
    /// lightmap, which must exist.
    ///
    /// This is used when constructing bind groups in non-bindless mode.
    pub(crate) fn bindings_for_first_lightmap(&self) -> (&TextureView, &Sampler) {
        (
            &self.lightmaps[0].gpu_image.texture_view,
            &self.lightmaps[0].gpu_image.sampler,
        )
    }
}

impl RenderLightmapsL {
    /// Creates a new slab, appends it to the end of the list, and returns its
    /// slab index.
    fn create_slab(&mut self, other: &mut RenderLightmapsU, fallback_images: &FallbackImage) -> LightmapSlabIndex {
        let slab_index = LightmapSlabIndex::from(self.slabs.len());
        other.free_slabs.grow_and_insert(slab_index.into());
        self.slabs
            .push(LightmapSlab::new(fallback_images, other.bindless_supported));
        slab_index
    }

    fn allocate(
        &mut self,
        other: &mut RenderLightmapsU,
        fallback_images: &FallbackImage,
        image_id: AssetId<Image>,
    ) -> (LightmapSlabIndex, LightmapSlotIndex) {
        let slab_index = match other.free_slabs.minimum() {
            None => self.create_slab(other, fallback_images),
            Some(slab_index) => slab_index.into(),
        };

        let slab = &mut self.slabs[usize::from(slab_index)];
        let slot_index = slab.allocate(image_id);
        if slab.is_full() {
            other.free_slabs.remove(slab_index.into());
        }

        (slab_index, slot_index)
    }

    fn remove(
        &mut self,
        other: &mut RenderLightmapsU,
        fallback_images: &FallbackImage,
        slab_index: LightmapSlabIndex,
        slot_index: LightmapSlotIndex,
    ) {
        let slab = &mut self.slabs[usize::from(slab_index)];
        slab.remove(fallback_images, slot_index);

        if !slab.is_full() {
            other.free_slabs.grow_and_insert(slot_index.into());
        }
    }
}
