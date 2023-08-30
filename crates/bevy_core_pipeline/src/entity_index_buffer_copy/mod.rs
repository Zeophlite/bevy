use bevy_app::Plugin;
use bevy_ecs::{query::QueryItem, world::World};
use bevy_render::{
    picking::{CurrentGpuPickingBufferIndex, ExtractedGpuPickingCamera, VisibleMeshIdTextures},
    render_graph::{RenderGraphApp, RenderGraphContext, ViewNode, ViewNodeRunner},
    renderer::RenderContext,
    RenderApp,
};

use crate::core_3d::graph::Core3d;

#[derive(Default)]
pub struct EntityIndexBufferCopyNode;
impl ViewNode for EntityIndexBufferCopyNode {
    type ViewQuery = (
        &'static VisibleMeshIdTextures,
        &'static ExtractedGpuPickingCamera,
    );

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        (mesh_id_textures, gpu_picking_camera): QueryItem<Self::ViewQuery>,
        world: &World,
    ) -> Result<(), bevy_render::render_graph::NodeRunError> {
        let current_buffer_index = world.resource::<CurrentGpuPickingBufferIndex>();
        gpu_picking_camera.run_node(
            render_context.command_encoder(),
            &mesh_id_textures.main.texture,
            current_buffer_index,
        );
        Ok(())
    }
}

pub struct EntityIndexBufferCopyPlugin;
impl Plugin for EntityIndexBufferCopyPlugin {
    fn build(&self, app: &mut bevy_app::App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        // 3D
        use crate::core_3d::graph::Node3d::*;
        render_app
            .add_render_graph_node::<ViewNodeRunner<EntityIndexBufferCopyNode>>(
                Core3d,
                EntityIndexBufferCopy,
            )
            .add_render_graph_edge(Core3d, Upscaling, EntityIndexBufferCopy);
    }
}
