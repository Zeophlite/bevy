use crate::{
    core_3d::Opaque3d,
    skybox::{SkyboxBindGroup, SkyboxPipelineId},
};
use bevy_ecs::{prelude::World, query::QueryItem};
use bevy_render::{
    camera::ExtractedCamera,
    diagnostic::RecordDiagnostics,
    picking::{ExtractedGpuPickingCamera, VisibleMeshIdTextures},
    render_phase::{BinnedRenderPhase, TrackedRenderPass},
    render_resource::{CommandEncoderDescriptor, StoreOp},
    render_graph::{NodeRunError, RenderGraphContext, ViewNode},
    render_phase::RenderPhase,
    render_resource::{
        LoadOp, Operations, PipelineCache, RenderPassColorAttachment,
        RenderPassDepthStencilAttachment, RenderPassDescriptor,
    },
    renderer::RenderContext,
    view::{ViewDepthTexture, ViewTarget, ViewUniformOffset},
};
#[cfg(feature = "trace")]
use bevy_utils::tracing::info_span;
use smallvec::{smallvec, SmallVec};

use super::AlphaMask3d;

/// A [`bevy_render::render_graph::Node`] that runs the [`Opaque3d`]
/// [`BinnedRenderPhase`] and [`AlphaMask3d`]
/// [`bevy_render::render_phase::SortedRenderPhase`]s.
#[derive(Default)]
pub struct MainOpaquePass3dNode;
impl ViewNode for MainOpaquePass3dNode {
    type ViewQuery = (
        &'static ExtractedCamera,
        &'static BinnedRenderPhase<Opaque3d>,
        &'static BinnedRenderPhase<AlphaMask3d>,
        &'static ViewTarget,
        &'static ViewDepthTexture,
        Option<&'static DepthPrepass>,
        Option<&'static NormalPrepass>,
        Option<&'static MotionVectorPrepass>,
        Option<&'static ExtractedGpuPickingCamera>,
        Option<&'static SkyboxPipelineId>,
        Option<&'static SkyboxBindGroup>,
        Option<&'static VisibleMeshIdTextures>,
        &'static ViewUniformOffset,
    );

    fn run<'w>(
        &self,
        graph: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        (
            camera,
            opaque_phase,
            alpha_mask_phase,
            target,
            depth,
            depth_prepass,
            normal_prepass,
            motion_vector_prepass,
            gpu_picking_camera,
            skybox_pipeline,
            skybox_bind_group,
            mesh_id_textures,
            view_uniform_offset,
        ): QueryItem<'w, Self::ViewQuery>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let diagnostics = render_context.diagnostic_recorder();

        let mut color_attachments: SmallVec<[Option<RenderPassColorAttachment>; 2]> =
            smallvec![Some(target.get_color_attachment(Operations {
                load: match camera_3d.clear_color {
                    ClearColorConfig::Default =>
                        LoadOp::Clear(world.resource::<ClearColor>().0.into()),
                    ClearColorConfig::Custom(color) => LoadOp::Clear(color.into()),
                    ClearColorConfig::None => LoadOp::Load,
                },
                store: true,
            }))];

        if gpu_picking_camera.is_some() {
            if let Some(mesh_id_textures) = mesh_id_textures {
                color_attachments.push(Some(mesh_id_textures.get_color_attachment(Operations {
                    load: match camera_3d.clear_color {
                        ClearColorConfig::None => LoadOp::Load,
                        // TODO clear this earlier?
                        _ => LoadOp::Clear(VisibleMeshIdTextures::CLEAR_COLOR),
                    },
                    store: true,
                })));
            }
        }

        // Setup render pass
        let mut render_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("main_opaque_pass_3d"),
            // NOTE: The opaque pass loads the color
            // buffer as well as writing to it.
            color_attachments: &color_attachments,
            depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                view: &depth.view,
                // NOTE: The opaque main pass loads the depth buffer and possibly overwrites it
                depth_ops: Some(Operations {
                    load: if depth_prepass.is_some()
                        || normal_prepass.is_some()
                        || motion_vector_prepass.is_some()
                    {
                        // if any prepass runs, it will generate a depth buffer so we should use it,
                        // even if only the normal_prepass is used.
                        Camera3dDepthLoadOp::Load
                    } else {
                        // NOTE: 0.0 is the far plane due to bevy's use of reverse-z projections.
                        camera_3d.depth_load_op.clone()
                    }
                    .into(),
                    store: true,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        if let Some(viewport) = camera.viewport.as_ref() {
            render_pass.set_camera_viewport(viewport);
        }

        let view_entity = graph.view_entity();
        render_context.add_command_buffer_generation_task(move |render_device| {
            #[cfg(feature = "trace")]
            let _main_opaque_pass_3d_span = info_span!("main_opaque_pass_3d").entered();

            // Command encoder setup
            let mut command_encoder =
                render_device.create_command_encoder(&CommandEncoderDescriptor {
                    label: Some("main_opaque_pass_3d_command_encoder"),
                });

            // Render pass setup
            let render_pass = command_encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("main_opaque_pass_3d"),
                color_attachments: &color_attachments,
                depth_stencil_attachment,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            let mut render_pass = TrackedRenderPass::new(&render_device, render_pass);
            let pass_span = diagnostics.pass_span(&mut render_pass, "main_opaque_pass_3d");

            if let Some(viewport) = camera.viewport.as_ref() {
                render_pass.set_camera_viewport(viewport);
            }

            // Opaque draws
            if !opaque_phase.is_empty() {
                #[cfg(feature = "trace")]
                let _opaque_main_pass_3d_span = info_span!("opaque_main_pass_3d").entered();
                opaque_phase.render(&mut render_pass, world, view_entity);
            }

            // Alpha draws
            if !alpha_mask_phase.is_empty() {
                #[cfg(feature = "trace")]
                let _alpha_mask_main_pass_3d_span = info_span!("alpha_mask_main_pass_3d").entered();
                alpha_mask_phase.render(&mut render_pass, world, view_entity);
            }

            // Skybox draw using a fullscreen triangle
            if let (Some(skybox_pipeline), Some(SkyboxBindGroup(skybox_bind_group))) =
                (skybox_pipeline, skybox_bind_group)
            {
                let pipeline_cache = world.resource::<PipelineCache>();
                if let Some(pipeline) = pipeline_cache.get_render_pipeline(skybox_pipeline.0) {
                    render_pass.set_render_pipeline(pipeline);
                    render_pass.set_bind_group(
                        0,
                        &skybox_bind_group.0,
                        &[view_uniform_offset.offset, skybox_bind_group.1],
                    );
                    render_pass.draw(0..3, 0..1);
                }
            }

            pass_span.end(&mut render_pass);
            drop(render_pass);
            command_encoder.finish()
        });

        Ok(())
    }
}
