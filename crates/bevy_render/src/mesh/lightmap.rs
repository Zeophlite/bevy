
use bevy_asset::AssetId;
use bevy_derive::{Deref, DerefMut};
use bevy_image::Image;
use fixedbitset::FixedBitSet;
use nonmax::{NonMaxU16, NonMaxU32};
use bevy_math::{uvec2, vec4, Rect, UVec2};
use bevy_ecs::{
    resource::Resource,
};
use bevy_platform::collections::HashSet;

use crate::sync_world::MainEntityHashMap;

/// The index of the slab (binding array) in which a lightmap is located.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Deref, DerefMut)]
#[repr(transparent)]
pub struct LightmapSlabIndex(pub NonMaxU32);

/// The index of the slot (element within the binding array) in the slab in
/// which a lightmap is located.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Deref, DerefMut)]
#[repr(transparent)]
pub struct LightmapSlotIndex(pub NonMaxU16);



/// Lightmap data stored in the render world.
///
/// There is one of these per visible lightmapped mesh instance.
#[derive(Debug)]
pub struct RenderLightmap {
    /// The rectangle within the lightmap texture that the UVs are relative to.
    ///
    /// The top left coordinate is the `min` part of the rect, and the bottom
    /// right coordinate is the `max` part of the rect. The rect ranges from (0,
    /// 0) to (1, 1).
    pub(crate) uv_rect: Rect,

    /// The index of the slab (i.e. binding array) in which the lightmap is
    /// located.
    pub slab_index: LightmapSlabIndex,

    /// The index of the slot (i.e. element within the binding array) in which
    /// the lightmap is located.
    ///
    /// If bindless lightmaps aren't in use, this will be 0.
    pub slot_index: LightmapSlotIndex,

    // Whether or not bicubic sampling should be used for this lightmap.
    pub bicubic_sampling: bool,
}

/// Stores data for all lightmaps in the render world.
///
/// This is cleared and repopulated each frame during the `extract_lightmaps`
/// system.
#[derive(Resource)]
pub struct RenderLightmapsU {
    /// The mapping from every lightmapped entity to its lightmap info.
    ///
    /// Entities without lightmaps, or for which the mesh or lightmap isn't
    /// loaded, won't have entries in this table.
    pub render_lightmaps: MainEntityHashMap<RenderLightmap>,

    /// The slabs (binding arrays) containing the lightmaps.
    pub slabs: Vec<LightmapSlabUnloaded>,

    pub free_slabs: FixedBitSet,

    pub pending_lightmaps: HashSet<(LightmapSlabIndex, LightmapSlotIndex)>,

    /// Whether bindless textures are supported on this platform.
    pub bindless_supported: bool,
}

/// A binding array that contains lightmaps.
///
/// This will have a single binding if bindless lightmaps aren't in use.
pub struct LightmapSlabUnloaded {
    /// The GPU images in this slab.
    pub lightmaps: Vec<AllocatedLightmapUnloaded>,
    pub free_slots_bitmask: u32,
}

pub struct AllocatedLightmapUnloaded {
    // This will only be present if the lightmap is allocated but not loaded.
    pub asset_id: AssetId<Image>,
}

/// Packs the lightmap UV rect into 64 bits (4 16-bit unsigned integers).
pub fn pack_lightmap_uv_rect(maybe_rect: Option<Rect>) -> UVec2 {
    match maybe_rect {
        Some(rect) => {
            let rect_uvec4 = (vec4(rect.min.x, rect.min.y, rect.max.x, rect.max.y) * 65535.0)
                .round()
                .as_uvec4();
            uvec2(
                rect_uvec4.x | (rect_uvec4.y << 16),
                rect_uvec4.z | (rect_uvec4.w << 16),
            )
        }
        None => UVec2::ZERO,
    }
}

impl RenderLightmap {
    /// Creates a new lightmap from a texture, a UV rect, and a slab and slot
    /// index pair.
    pub fn new(
        uv_rect: Rect,
        slab_index: LightmapSlabIndex,
        slot_index: LightmapSlotIndex,
        bicubic_sampling: bool,
    ) -> Self {
        Self {
            uv_rect,
            slab_index,
            slot_index,
            bicubic_sampling,
        }
    }
}

impl From<u32> for LightmapSlabIndex {
    fn from(value: u32) -> Self {
        Self(NonMaxU32::new(value).unwrap())
    }
}

impl From<usize> for LightmapSlabIndex {
    fn from(value: usize) -> Self {
        Self::from(value as u32)
    }
}

impl From<u32> for LightmapSlotIndex {
    fn from(value: u32) -> Self {
        Self(NonMaxU16::new(value as u16).unwrap())
    }
}

impl From<usize> for LightmapSlotIndex {
    fn from(value: usize) -> Self {
        Self::from(value as u32)
    }
}

impl From<LightmapSlabIndex> for usize {
    fn from(value: LightmapSlabIndex) -> Self {
        value.0.get() as usize
    }
}

impl From<LightmapSlotIndex> for usize {
    fn from(value: LightmapSlotIndex) -> Self {
        value.0.get() as usize
    }
}

impl From<LightmapSlotIndex> for u16 {
    fn from(value: LightmapSlotIndex) -> Self {
        value.0.get()
    }
}

impl From<LightmapSlotIndex> for u32 {
    fn from(value: LightmapSlotIndex) -> Self {
        value.0.get() as u32
    }
}
