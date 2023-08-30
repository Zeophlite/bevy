use crate::core_3d::Transparent3d;
use bevy_ecs::{prelude::*, query::QueryItem};
use bevy_render::{
    camera::ExtractedCamera,
    diagnostic::RecordDiagnostics,
    picking::{ExtractedGpuPickingCamera, VisibleMeshIdTextures},
    render_graph::{NodeRunError, RenderGraphContext, ViewNode},
    render_phase::ViewSortedRenderPhases,
    render_phase::SortedRenderPhase,
    render_resource::{LoadOp, Operations, RenderPassDepthStencilAttachment, RenderPassDescriptor, StoreOp},
    renderer::RenderContext,
    view::{ViewDepthTexture, ViewTarget},
};
#[cfg(feature = "trace")]
use bevy_utils::tracing::info_span;

/// A [`bevy_render::render_graph::Node`] that runs the [`Transparent3d`]
/// [`SortedRenderPhase`].
#[derive(Default)]
pub struct MainTransparentPass3dNode;

impl ViewNode for MainTransparentPass3dNode {
    type ViewQuery = (
        &'static ExtractedCamera,
        &'static ViewTarget,
        &'static ViewDepthTexture,
        Option<&'static ExtractedGpuPickingCamera>,
        Option<&'static VisibleMeshIdTextures>,
    );
    fn run(
        &self,
        graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        (camera, target, depth, gpu_picking_camera, mesh_id_textures): QueryItem<
            Self::ViewQuery,
        >,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let view_entity = graph.view_entity();

        let Some(transparent_phases) =
            world.get_resource::<ViewSortedRenderPhases<Transparent3d>>()
        else {
            return Ok(());
        };

        let Some(transparent_phase) = transparent_phases.get(&view_entity) else {
            return Ok(());
        };

        if !transparent_phase.items.is_empty() {
            // Run the transparent pass, sorted back-to-front
            // NOTE: Scoped to drop the mutable borrow of render_context
            #[cfg(feature = "trace")]
            let _main_transparent_pass_3d_span = info_span!("main_transparent_pass_3d").entered();

            let diagnostics = render_context.diagnostic_recorder();

            // NOTE: The transparent pass loads the color buffer as well as overwriting it where appropriate.
            let mut color_attachments = vec![Some(target.get_color_attachment())];

            if gpu_picking_camera.is_some() {
                if let Some(mesh_id_textures) = mesh_id_textures {
                    color_attachments.push(Some(mesh_id_textures.get_color_attachment(
                        Operations {
                            // The texture is already cleared in the opaque pass
                            load: LoadOp::Load,
                            store: StoreOp::Store,
                        },
                    )));
                }
            }

            // TODO_DS: which one
            let mut render_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
                label: Some("main_transparent_pass_3d"),
                // color_attachments: &[Some(target.get_color_attachment())],
                color_attachments: &color_attachments,
                // depth_stencil_attachment: Some(depth.get_attachment(StoreOp::Store)),
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                    view: &depth.view(),
                    // NOTE: For the transparent pass we load the depth buffer. There should be no
                    // need to write to it, but store is set to `true` as a workaround for issue #3776,
                    // https://github.com/bevyengine/bevy/issues/3776
                    // so that wgpu does not clear the depth buffer.
                    // As the opaque and alpha mask passes run first, opaque meshes can occlude
                    // transparent ones.
                    depth_ops: Some(Operations {
                        load: LoadOp::Load,
                        store: StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            let pass_span = diagnostics.pass_span(&mut render_pass, "main_transparent_pass_3d");

            if let Some(viewport) = camera.viewport.as_ref() {
                render_pass.set_camera_viewport(viewport);
            }

            transparent_phase.render(&mut render_pass, world, view_entity);

            pass_span.end(&mut render_pass);
        }

        // WebGL2 quirk: if ending with a render pass with a custom viewport, the viewport isn't
        // reset for the next render pass so add an empty render pass without a custom viewport
        #[cfg(all(feature = "webgl", target_arch = "wasm32", not(feature = "webgpu")))]
        if camera.viewport.is_some() {
            #[cfg(feature = "trace")]
            let _reset_viewport_pass_3d = info_span!("reset_viewport_pass_3d").entered();
            let pass_descriptor = RenderPassDescriptor {
                label: Some("reset_viewport_pass_3d"),
                color_attachments: &[Some(target.get_color_attachment())],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            };

            render_context
                .command_encoder()
                .begin_render_pass(&pass_descriptor);
        }

        Ok(())
    }
}
