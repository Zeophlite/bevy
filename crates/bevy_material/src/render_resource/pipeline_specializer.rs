use crate::render_resource::{ComputePipelineDescriptor, RenderPipelineDescriptor};
use bevy_mesh::{MeshVertexBufferLayoutRef, MissingVertexAttributeError};
use core::{fmt::Debug, hash::Hash};
use thiserror::Error;
use tracing::error;

/// A trait that allows constructing different variants of a render pipeline from a key.
///
/// Note: This is intended for modifying your pipeline descriptor on the basis of a key. If your key
/// contains no data then you don't need to specialize. For example, if you are using the
/// [`AsBindGroup`](crate::render_resource::AsBindGroup) without the `#[bind_group_data]` attribute,
/// you don't need to specialize. Instead, create the pipeline directly from [`PipelineCache`] and
/// store its ID.
///
/// See [`SpecializedRenderPipelines`] for more info.
pub trait SpecializedRenderPipeline {
    /// The key that defines each "variant" of the render pipeline.
    type Key: Clone + Hash + PartialEq + Eq;

    /// Construct a new render pipeline based on the provided key.
    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor;
}

/// A trait that allows constructing different variants of a compute pipeline from a key.
///
/// Note: This is intended for modifying your pipeline descriptor on the basis of a key. If your key
/// contains no data then you don't need to specialize. For example, if you are using the
/// [`AsBindGroup`](crate::render_resource::AsBindGroup) without the `#[bind_group_data]` attribute,
/// you don't need to specialize. Instead, create the pipeline directly from [`PipelineCache`] and
/// store its ID.
///
/// See [`SpecializedComputePipelines`] for more info.
pub trait SpecializedComputePipeline {
    /// The key that defines each "variant" of the compute pipeline.
    type Key: Clone + Hash + PartialEq + Eq;

    /// Construct a new compute pipeline based on the provided key.
    fn specialize(&self, key: Self::Key) -> ComputePipelineDescriptor;
}

/// A trait that allows constructing different variants of a render pipeline from a key and the
/// particular mesh's vertex buffer layout.
///
/// See [`SpecializedMeshPipelines`] for more info.
pub trait SpecializedMeshPipeline {
    /// The key that defines each "variant" of the render pipeline.
    type Key: Clone + Hash + PartialEq + Eq;

    /// Construct a new render pipeline based on the provided key and vertex layout.
    ///
    /// The returned pipeline descriptor should have a single vertex buffer, which is derived from
    /// `layout`.
    fn specialize(
        &self,
        key: Self::Key,
        layout: &MeshVertexBufferLayoutRef,
    ) -> Result<RenderPipelineDescriptor, SpecializedMeshPipelineError>;
}

#[derive(Error, Debug)]
pub enum SpecializedMeshPipelineError {
    #[error(transparent)]
    MissingVertexAttribute(#[from] MissingVertexAttributeError),
}
