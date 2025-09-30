use crate::render_resource::BindGroupLayoutDescriptor;

/// All possible [`BindGroupLayout`]s in bevy's default mesh shader (`mesh.wgsl`).
#[derive(Clone)]
pub struct MeshLayouts {
    /// The mesh model uniform (transform) and nothing else.
    pub model_only: BindGroupLayoutDescriptor,

    /// Includes the lightmap texture and uniform.
    pub lightmapped: BindGroupLayoutDescriptor,

    /// Also includes the uniform for skinning
    pub skinned: BindGroupLayoutDescriptor,

    /// Like [`MeshLayouts::skinned`], but includes slots for the previous
    /// frame's joint matrices, so that we can compute motion vectors.
    pub skinned_motion: BindGroupLayoutDescriptor,

    /// Also includes the uniform and [`MorphAttributes`] for morph targets.
    ///
    /// [`MorphAttributes`]: bevy_mesh::morph::MorphAttributes
    pub morphed: BindGroupLayoutDescriptor,

    /// Like [`MeshLayouts::morphed`], but includes a slot for the previous
    /// frame's morph weights, so that we can compute motion vectors.
    pub morphed_motion: BindGroupLayoutDescriptor,

    /// Also includes both uniforms for skinning and morph targets, also the
    /// morph target [`MorphAttributes`] binding.
    ///
    /// [`MorphAttributes`]: bevy_mesh::morph::MorphAttributes
    pub morphed_skinned: BindGroupLayoutDescriptor,

    /// Like [`MeshLayouts::morphed_skinned`], but includes slots for the
    /// previous frame's joint matrices and morph weights, so that we can
    /// compute motion vectors.
    pub morphed_skinned_motion: BindGroupLayoutDescriptor,
}

impl MeshLayouts {
    /// Prepare the layouts used by the default bevy [`Mesh`].
    ///
    /// [`Mesh`]: bevy_mesh::Mesh
    pub fn new(render_device: &RenderDevice, render_adapter: &RenderAdapter) -> Self {
        MeshLayouts {
            model_only: Self::model_only_layout(render_device),
            lightmapped: Self::lightmapped_layout(render_device, render_adapter),
            skinned: Self::skinned_layout(render_device),
            skinned_motion: Self::skinned_motion_layout(render_device),
            morphed: Self::morphed_layout(render_device),
            morphed_motion: Self::morphed_motion_layout(render_device),
            morphed_skinned: Self::morphed_skinned_layout(render_device),
            morphed_skinned_motion: Self::morphed_skinned_motion_layout(render_device),
        }
    }

    // ---------- create individual BindGroupLayouts ----------

    fn model_only_layout(render_device: &RenderDevice) -> BindGroupLayoutDescriptor {
        BindGroupLayoutDescriptor::new(
            "mesh_layout",
            &BindGroupLayoutEntries::single(
                ShaderStages::empty(),
                layout_entry::model(render_device),
            ),
        )
    }

    /// Creates the layout for skinned meshes.
    fn skinned_layout(render_device: &RenderDevice) -> BindGroupLayoutDescriptor {
        BindGroupLayoutDescriptor::new(
            "skinned_mesh_layout",
            &BindGroupLayoutEntries::with_indices(
                ShaderStages::VERTEX,
                (
                    (0, layout_entry::model(render_device)),
                    // The current frame's joint matrix buffer.
                    (1, layout_entry::skinning(render_device)),
                ),
            ),
        )
    }

    /// Creates the layout for skinned meshes with the infrastructure to compute
    /// motion vectors.
    fn skinned_motion_layout(render_device: &RenderDevice) -> BindGroupLayoutDescriptor {
        BindGroupLayoutDescriptor::new(
            "skinned_motion_mesh_layout",
            &BindGroupLayoutEntries::with_indices(
                ShaderStages::VERTEX,
                (
                    (0, layout_entry::model(render_device)),
                    // The current frame's joint matrix buffer.
                    (1, layout_entry::skinning(render_device)),
                    // The previous frame's joint matrix buffer.
                    (6, layout_entry::skinning(render_device)),
                ),
            ),
        )
    }

    /// Creates the layout for meshes with morph targets.
    fn morphed_layout(render_device: &RenderDevice) -> BindGroupLayoutDescriptor {
        BindGroupLayoutDescriptor::new(
            "morphed_mesh_layout",
            &BindGroupLayoutEntries::with_indices(
                ShaderStages::VERTEX,
                (
                    (0, layout_entry::model(render_device)),
                    // The current frame's morph weight buffer.
                    (2, layout_entry::weights()),
                    (3, layout_entry::targets()),
                ),
            ),
        )
    }

    /// Creates the layout for meshes with morph targets and the infrastructure
    /// to compute motion vectors.
    fn morphed_motion_layout(render_device: &RenderDevice) -> BindGroupLayoutDescriptor {
        BindGroupLayoutDescriptor::new(
            "morphed_mesh_layout",
            &BindGroupLayoutEntries::with_indices(
                ShaderStages::VERTEX,
                (
                    (0, layout_entry::model(render_device)),
                    // The current frame's morph weight buffer.
                    (2, layout_entry::weights()),
                    (3, layout_entry::targets()),
                    // The previous frame's morph weight buffer.
                    (7, layout_entry::weights()),
                ),
            ),
        )
    }

    /// Creates the bind group layout for meshes with both skins and morph
    /// targets.
    fn morphed_skinned_layout(render_device: &RenderDevice) -> BindGroupLayoutDescriptor {
        BindGroupLayoutDescriptor::new(
            "morphed_skinned_mesh_layout",
            &BindGroupLayoutEntries::with_indices(
                ShaderStages::VERTEX,
                (
                    (0, layout_entry::model(render_device)),
                    // The current frame's joint matrix buffer.
                    (1, layout_entry::skinning(render_device)),
                    // The current frame's morph weight buffer.
                    (2, layout_entry::weights()),
                    (3, layout_entry::targets()),
                ),
            ),
        )
    }

    /// Creates the bind group layout for meshes with both skins and morph
    /// targets, in addition to the infrastructure to compute motion vectors.
    fn morphed_skinned_motion_layout(render_device: &RenderDevice) -> BindGroupLayoutDescriptor {
        BindGroupLayoutDescriptor::new(
            "morphed_skinned_motion_mesh_layout",
            &BindGroupLayoutEntries::with_indices(
                ShaderStages::VERTEX,
                (
                    (0, layout_entry::model(render_device)),
                    // The current frame's joint matrix buffer.
                    (1, layout_entry::skinning(render_device)),
                    // The current frame's morph weight buffer.
                    (2, layout_entry::weights()),
                    (3, layout_entry::targets()),
                    // The previous frame's joint matrix buffer.
                    (6, layout_entry::skinning(render_device)),
                    // The previous frame's morph weight buffer.
                    (7, layout_entry::weights()),
                ),
            ),
        )
    }

    fn lightmapped_layout(
        render_device: &RenderDevice,
        render_adapter: &RenderAdapter,
    ) -> BindGroupLayoutDescriptor {
        if binding_arrays_are_usable(render_device, render_adapter) {
            BindGroupLayoutDescriptor::new(
                "lightmapped_mesh_layout",
                &BindGroupLayoutEntries::with_indices(
                    ShaderStages::VERTEX,
                    (
                        (0, layout_entry::model(render_device)),
                        (4, layout_entry::lightmaps_texture_view_array()),
                        (5, layout_entry::lightmaps_sampler_array()),
                    ),
                ),
            )
        } else {
            BindGroupLayoutDescriptor::new(
                "lightmapped_mesh_layout",
                &BindGroupLayoutEntries::with_indices(
                    ShaderStages::VERTEX,
                    (
                        (0, layout_entry::model(render_device)),
                        (4, layout_entry::lightmaps_texture_view()),
                        (5, layout_entry::lightmaps_sampler()),
                    ),
                ),
            )
        }
    }

    // ---------- BindGroup methods ----------

    pub fn model_only(
        &self,
        render_device: &RenderDevice,
        pipeline_cache: &PipelineCache,
        model: &BindingResource,
    ) -> BindGroup {
        render_device.create_bind_group(
            "model_only_mesh_bind_group",
            &pipeline_cache.get_bind_group_layout(&self.model_only),
            &[entry::model(0, model.clone())],
        )
    }

    pub fn lightmapped(
        &self,
        render_device: &RenderDevice,
        pipeline_cache: &PipelineCache,
        model: &BindingResource,
        lightmap_slab: &LightmapSlab,
        bindless_lightmaps: bool,
    ) -> BindGroup {
        if bindless_lightmaps {
            let (texture_views, samplers) = lightmap_slab.build_binding_arrays();
            render_device.create_bind_group(
                "lightmapped_mesh_bind_group",
                &pipeline_cache.get_bind_group_layout(&self.lightmapped),
                &[
                    entry::model(0, model.clone()),
                    entry::lightmaps_texture_view_array(4, &texture_views),
                    entry::lightmaps_sampler_array(5, &samplers),
                ],
            )
        } else {
            let (texture_view, sampler) = lightmap_slab.bindings_for_first_lightmap();
            render_device.create_bind_group(
                "lightmapped_mesh_bind_group",
                &pipeline_cache.get_bind_group_layout(&self.lightmapped),
                &[
                    entry::model(0, model.clone()),
                    entry::lightmaps_texture_view(4, texture_view),
                    entry::lightmaps_sampler(5, sampler),
                ],
            )
        }
    }

    /// Creates the bind group for skinned meshes with no morph targets.
    pub fn skinned(
        &self,
        render_device: &RenderDevice,
        pipeline_cache: &PipelineCache,
        model: &BindingResource,
        current_skin: &Buffer,
    ) -> BindGroup {
        render_device.create_bind_group(
            "skinned_mesh_bind_group",
            &pipeline_cache.get_bind_group_layout(&self.skinned),
            &[
                entry::model(0, model.clone()),
                entry::skinning(render_device, 1, current_skin),
            ],
        )
    }

    /// Creates the bind group for skinned meshes with no morph targets, with
    /// the infrastructure to compute motion vectors.
    ///
    /// `current_skin` is the buffer of joint matrices for this frame;
    /// `prev_skin` is the buffer for the previous frame. The latter is used for
    /// motion vector computation. If there is no such applicable buffer,
    /// `current_skin` and `prev_skin` will reference the same buffer.
    pub fn skinned_motion(
        &self,
        render_device: &RenderDevice,
        pipeline_cache: &PipelineCache,
        model: &BindingResource,
        current_skin: &Buffer,
        prev_skin: &Buffer,
    ) -> BindGroup {
        render_device.create_bind_group(
            "skinned_motion_mesh_bind_group",
            &pipeline_cache.get_bind_group_layout(&self.skinned_motion),
            &[
                entry::model(0, model.clone()),
                entry::skinning(render_device, 1, current_skin),
                entry::skinning(render_device, 6, prev_skin),
            ],
        )
    }

    /// Creates the bind group for meshes with no skins but morph targets.
    pub fn morphed(
        &self,
        render_device: &RenderDevice,
        pipeline_cache: &PipelineCache,
        model: &BindingResource,
        current_weights: &Buffer,
        targets: &TextureView,
    ) -> BindGroup {
        render_device.create_bind_group(
            "morphed_mesh_bind_group",
            &pipeline_cache.get_bind_group_layout(&self.morphed),
            &[
                entry::model(0, model.clone()),
                entry::weights(2, current_weights),
                entry::targets(3, targets),
            ],
        )
    }

    /// Creates the bind group for meshes with no skins but morph targets, in
    /// addition to the infrastructure to compute motion vectors.
    ///
    /// `current_weights` is the buffer of morph weights for this frame;
    /// `prev_weights` is the buffer for the previous frame. The latter is used
    /// for motion vector computation. If there is no such applicable buffer,
    /// `current_weights` and `prev_weights` will reference the same buffer.
    pub fn morphed_motion(
        &self,
        render_device: &RenderDevice,
        pipeline_cache: &PipelineCache,
        model: &BindingResource,
        current_weights: &Buffer,
        targets: &TextureView,
        prev_weights: &Buffer,
    ) -> BindGroup {
        render_device.create_bind_group(
            "morphed_motion_mesh_bind_group",
            &pipeline_cache.get_bind_group_layout(&self.morphed_motion),
            &[
                entry::model(0, model.clone()),
                entry::weights(2, current_weights),
                entry::targets(3, targets),
                entry::weights(7, prev_weights),
            ],
        )
    }

    /// Creates the bind group for meshes with skins and morph targets.
    pub fn morphed_skinned(
        &self,
        render_device: &RenderDevice,
        pipeline_cache: &PipelineCache,
        model: &BindingResource,
        current_skin: &Buffer,
        current_weights: &Buffer,
        targets: &TextureView,
    ) -> BindGroup {
        render_device.create_bind_group(
            "morphed_skinned_mesh_bind_group",
            &pipeline_cache.get_bind_group_layout(&self.morphed_skinned),
            &[
                entry::model(0, model.clone()),
                entry::skinning(render_device, 1, current_skin),
                entry::weights(2, current_weights),
                entry::targets(3, targets),
            ],
        )
    }

    /// Creates the bind group for meshes with skins and morph targets, in
    /// addition to the infrastructure to compute motion vectors.
    ///
    /// See the documentation for [`MeshLayouts::skinned_motion`] and
    /// [`MeshLayouts::morphed_motion`] above for more information about the
    /// `current_skin`, `prev_skin`, `current_weights`, and `prev_weights`
    /// buffers.
    pub fn morphed_skinned_motion(
        &self,
        render_device: &RenderDevice,
        pipeline_cache: &PipelineCache,
        model: &BindingResource,
        current_skin: &Buffer,
        current_weights: &Buffer,
        targets: &TextureView,
        prev_skin: &Buffer,
        prev_weights: &Buffer,
    ) -> BindGroup {
        render_device.create_bind_group(
            "morphed_skinned_motion_mesh_bind_group",
            &pipeline_cache.get_bind_group_layout(&self.morphed_skinned_motion),
            &[
                entry::model(0, model.clone()),
                entry::skinning(render_device, 1, current_skin),
                entry::weights(2, current_weights),
                entry::targets(3, targets),
                entry::skinning(render_device, 6, prev_skin),
                entry::weights(7, prev_weights),
            ],
        )
    }
}
