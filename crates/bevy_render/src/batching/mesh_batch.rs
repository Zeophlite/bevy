use bevy_asset::AssetId;
use bevy_ecs::{entity::Entity, system::{lifetimeless::SRes, SystemParamItem}};
use bevy_material::render::MeshPipeline;
use bevy_mesh::Mesh;
use bevy_utils::default;
use nonmax::NonMaxU32;
use tracing::{error, warn};

use crate::{
    batching::{gpu_preprocessing::{IndirectParametersCpuMetadata, UntypedPhaseIndirectParametersBuffers}, GetBatchData, GetFullBatchData},
    mesh::{allocator::MeshAllocator, lightmap::{LightmapSlabIndex, RenderLightmapsU}, material_bind_group::MaterialBindGroupIndex, mesh::{
        MeshInputUniform,
        MeshUniform,
        RenderMeshInstances
    }, skin::SkinUniforms, RenderMesh},
    render_asset::RenderAssets, sync_world::MainEntity
};

impl GetBatchData for MeshPipeline {
    type Param = (
        SRes<RenderMeshInstances>,
        SRes<RenderLightmapsU>,
        SRes<RenderAssets<RenderMesh>>,
        SRes<MeshAllocator>,
        SRes<SkinUniforms>,
    );
    // The material bind group ID, the mesh ID, and the lightmap ID,
    // respectively.
    type CompareData = (
        MaterialBindGroupIndex,
        AssetId<Mesh>,
        Option<LightmapSlabIndex>,
    );

    type BufferData = MeshUniform;

    fn get_batch_data(
        (mesh_instances, lightmaps, _, mesh_allocator, skin_uniforms): &SystemParamItem<
            Self::Param,
        >,
        (_entity, main_entity): (Entity, MainEntity),
    ) -> Option<(Self::BufferData, Option<Self::CompareData>)> {
        let RenderMeshInstances::CpuBuilding(ref mesh_instances) = **mesh_instances else {
            error!(
                "`get_batch_data` should never be called in GPU mesh uniform \
                building mode"
            );
            return None;
        };
        let mesh_instance = mesh_instances.get(&main_entity)?;
        let first_vertex_index =
            match mesh_allocator.mesh_vertex_slice(&mesh_instance.mesh_asset_id) {
                Some(mesh_vertex_slice) => mesh_vertex_slice.range.start,
                None => 0,
            };
        let maybe_lightmap = lightmaps.render_lightmaps.get(&main_entity);

        let current_skin_index = skin_uniforms.skin_index(main_entity);
        let material_bind_group_index = mesh_instance.material_bindings_index;

        Some((
            MeshUniform::new(
                &mesh_instance.transforms,
                first_vertex_index,
                material_bind_group_index.slot,
                maybe_lightmap.map(|lightmap| (lightmap.slot_index, lightmap.uv_rect)),
                current_skin_index,
                Some(mesh_instance.tag),
            ),
            mesh_instance.should_batch().then_some((
                material_bind_group_index.group,
                mesh_instance.mesh_asset_id,
                maybe_lightmap.map(|lightmap| lightmap.slab_index),
            )),
        ))
    }
}

impl GetFullBatchData for MeshPipeline {
    type BufferInputData = MeshInputUniform;

    fn get_index_and_compare_data(
        (mesh_instances, lightmaps, _, _, _): &SystemParamItem<Self::Param>,
        main_entity: MainEntity,
    ) -> Option<(NonMaxU32, Option<Self::CompareData>)> {
        // This should only be called during GPU building.
        let RenderMeshInstances::GpuBuilding(ref mesh_instances) = **mesh_instances else {
            error!(
                "`get_index_and_compare_data` should never be called in CPU mesh uniform building \
                mode"
            );
            return None;
        };

        let mesh_instance = mesh_instances.get(&main_entity)?;
        let maybe_lightmap = lightmaps.render_lightmaps.get(&main_entity);

        Some((
            mesh_instance.current_uniform_index,
            mesh_instance.should_batch().then_some((
                mesh_instance.material_bindings_index.group,
                mesh_instance.mesh_asset_id,
                maybe_lightmap.map(|lightmap| lightmap.slab_index),
            )),
        ))
    }

    fn get_binned_batch_data(
        (mesh_instances, lightmaps, _, mesh_allocator, skin_uniforms): &SystemParamItem<
            Self::Param,
        >,
        main_entity: MainEntity,
    ) -> Option<Self::BufferData> {
        let RenderMeshInstances::CpuBuilding(ref mesh_instances) = **mesh_instances else {
            error!(
                "`get_binned_batch_data` should never be called in GPU mesh uniform building mode"
            );
            return None;
        };
        let mesh_instance = mesh_instances.get(&main_entity)?;
        let first_vertex_index =
            match mesh_allocator.mesh_vertex_slice(&mesh_instance.mesh_asset_id) {
                Some(mesh_vertex_slice) => mesh_vertex_slice.range.start,
                None => 0,
            };
        let maybe_lightmap = lightmaps.render_lightmaps.get(&main_entity);

        let current_skin_index = skin_uniforms.skin_index(main_entity);

        Some(MeshUniform::new(
            &mesh_instance.transforms,
            first_vertex_index,
            mesh_instance.material_bindings_index.slot,
            maybe_lightmap.map(|lightmap| (lightmap.slot_index, lightmap.uv_rect)),
            current_skin_index,
            Some(mesh_instance.tag),
        ))
    }

    fn get_binned_index(
        (mesh_instances, _, _, _, _): &SystemParamItem<Self::Param>,
        main_entity: MainEntity,
    ) -> Option<NonMaxU32> {
        // This should only be called during GPU building.
        let RenderMeshInstances::GpuBuilding(ref mesh_instances) = **mesh_instances else {
            error!(
                "`get_binned_index` should never be called in CPU mesh uniform \
                building mode"
            );
            return None;
        };

        mesh_instances
            .get(&main_entity)
            .map(|entity| entity.current_uniform_index)
    }

    fn write_batch_indirect_parameters_metadata(
        indexed: bool,
        base_output_index: u32,
        batch_set_index: Option<NonMaxU32>,
        phase_indirect_parameters_buffers: &mut UntypedPhaseIndirectParametersBuffers,
        indirect_parameters_offset: u32,
    ) {
        let indirect_parameters = IndirectParametersCpuMetadata {
            base_output_index,
            batch_set_index: match batch_set_index {
                Some(batch_set_index) => u32::from(batch_set_index),
                None => !0,
            },
        };

        if indexed {
            phase_indirect_parameters_buffers
                .indexed
                .set(indirect_parameters_offset, indirect_parameters);
        } else {
            phase_indirect_parameters_buffers
                .non_indexed
                .set(indirect_parameters_offset, indirect_parameters);
        }
    }
}


