//! Bind group layout related definitions for the mesh pipeline.

use bevy_material::{
    render::MeshLayouts,
    render_resource::{BindGroupLayoutDescriptor, BindGroupLayoutEntries, ShaderStages},
};
use bevy_math::Mat4;
use bevy_mesh::morph::MAX_MORPH_WEIGHTS;
use bevy_render::{
    render_resource::*,
    renderer::{RenderAdapter, RenderDevice},
};

use crate::{binding_arrays_are_usable, render::skin::MAX_JOINTS, LightmapSlab};

const MORPH_WEIGHT_SIZE: usize = size_of::<f32>();

/// This is used to allocate buffers.
/// The correctness of the value depends on the GPU/platform.
/// The current value is chosen because it is guaranteed to work everywhere.
/// To allow for bigger values, a check must be made for the limits
/// of the GPU at runtime, which would mean not using consts anymore.
pub const MORPH_BUFFER_SIZE: usize = MAX_MORPH_WEIGHTS * MORPH_WEIGHT_SIZE;

const JOINT_SIZE: usize = size_of::<Mat4>();
pub(crate) const JOINT_BUFFER_SIZE: usize = MAX_JOINTS * JOINT_SIZE;

/// Individual layout entries.
mod layout_entry {
    use core::num::NonZeroU32;

    use super::{JOINT_BUFFER_SIZE, MORPH_BUFFER_SIZE};
    use crate::{render::skin, MeshUniform, LIGHTMAPS_PER_SLAB};
    use bevy_material::render_resource::{
        binding_types::{
            sampler, storage_buffer_read_only_sized, texture_2d, texture_3d, uniform_buffer_sized,
        },
        BindGroupLayoutEntryBuilder, BufferSize, SamplerBindingType, ShaderStages,
        TextureSampleType,
    };
    use bevy_render::{render_resource::GpuArrayBuffer, renderer::RenderDevice};

    pub(super) fn model(render_device: &RenderDevice) -> BindGroupLayoutEntryBuilder {
        GpuArrayBuffer::<MeshUniform>::binding_layout(render_device)
            .visibility(ShaderStages::VERTEX_FRAGMENT)
    }
    pub(super) fn skinning(render_device: &RenderDevice) -> BindGroupLayoutEntryBuilder {
        // If we can use storage buffers, do so. Otherwise, fall back to uniform
        // buffers.
        let size = BufferSize::new(JOINT_BUFFER_SIZE as u64);
        if skin::skins_use_uniform_buffers(render_device) {
            uniform_buffer_sized(true, size)
        } else {
            storage_buffer_read_only_sized(false, size)
        }
    }
    pub(super) fn weights() -> BindGroupLayoutEntryBuilder {
        uniform_buffer_sized(true, BufferSize::new(MORPH_BUFFER_SIZE as u64))
    }
    pub(super) fn targets() -> BindGroupLayoutEntryBuilder {
        texture_3d(TextureSampleType::Float { filterable: false })
    }
    pub(super) fn lightmaps_texture_view() -> BindGroupLayoutEntryBuilder {
        texture_2d(TextureSampleType::Float { filterable: true }).visibility(ShaderStages::FRAGMENT)
    }
    pub(super) fn lightmaps_sampler() -> BindGroupLayoutEntryBuilder {
        sampler(SamplerBindingType::Filtering).visibility(ShaderStages::FRAGMENT)
    }
    pub(super) fn lightmaps_texture_view_array() -> BindGroupLayoutEntryBuilder {
        texture_2d(TextureSampleType::Float { filterable: true })
            .visibility(ShaderStages::FRAGMENT)
            .count(NonZeroU32::new(LIGHTMAPS_PER_SLAB as u32).unwrap())
    }
    pub(super) fn lightmaps_sampler_array() -> BindGroupLayoutEntryBuilder {
        sampler(SamplerBindingType::Filtering)
            .visibility(ShaderStages::FRAGMENT)
            .count(NonZeroU32::new(LIGHTMAPS_PER_SLAB as u32).unwrap())
    }
}

/// Individual [`BindGroupEntry`]
/// for bind groups.
mod entry {
    use crate::render::skin;

    use super::{JOINT_BUFFER_SIZE, MORPH_BUFFER_SIZE};
    use bevy_material::render_resource::BufferSize;
    use bevy_render::{
        render_resource::{
            BindGroupEntry, BindingResource, Buffer, BufferBinding, Sampler, TextureView,
            WgpuSampler, WgpuTextureView,
        },
        renderer::RenderDevice,
    };

    fn entry(binding: u32, size: Option<u64>, buffer: &Buffer) -> BindGroupEntry<'_> {
        BindGroupEntry {
            binding,
            resource: BindingResource::Buffer(BufferBinding {
                buffer,
                offset: 0,
                size: size.map(|size| BufferSize::new(size).unwrap()),
            }),
        }
    }
    pub(super) fn model(binding: u32, resource: BindingResource) -> BindGroupEntry {
        BindGroupEntry { binding, resource }
    }
    pub(super) fn skinning<'a>(
        render_device: &RenderDevice,
        binding: u32,
        buffer: &'a Buffer,
    ) -> BindGroupEntry<'a> {
        let size = if skin::skins_use_uniform_buffers(render_device) {
            Some(JOINT_BUFFER_SIZE as u64)
        } else {
            None
        };
        entry(binding, size, buffer)
    }
    pub(super) fn weights(binding: u32, buffer: &Buffer) -> BindGroupEntry<'_> {
        entry(binding, Some(MORPH_BUFFER_SIZE as u64), buffer)
    }
    pub(super) fn targets(binding: u32, texture: &TextureView) -> BindGroupEntry<'_> {
        BindGroupEntry {
            binding,
            resource: BindingResource::TextureView(texture),
        }
    }
    pub(super) fn lightmaps_texture_view(
        binding: u32,
        texture: &TextureView,
    ) -> BindGroupEntry<'_> {
        BindGroupEntry {
            binding,
            resource: BindingResource::TextureView(texture),
        }
    }
    pub(super) fn lightmaps_sampler(binding: u32, sampler: &Sampler) -> BindGroupEntry<'_> {
        BindGroupEntry {
            binding,
            resource: BindingResource::Sampler(sampler),
        }
    }
    pub(super) fn lightmaps_texture_view_array<'a>(
        binding: u32,
        textures: &'a [&'a WgpuTextureView],
    ) -> BindGroupEntry<'a> {
        BindGroupEntry {
            binding,
            resource: BindingResource::TextureViewArray(textures),
        }
    }
    pub(super) fn lightmaps_sampler_array<'a>(
        binding: u32,
        samplers: &'a [&'a WgpuSampler],
    ) -> BindGroupEntry<'a> {
        BindGroupEntry {
            binding,
            resource: BindingResource::SamplerArray(samplers),
        }
    }
}
