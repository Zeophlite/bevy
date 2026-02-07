//! # Useful Environment Variables
//!
//! Both `bevy_render` and `wgpu` have a number of environment variable options for changing the runtime behavior
//! of both crates. Many of these may be useful in development or release environments.
//!
//! - `WGPU_DEBUG=1` enables debug labels, which can be useful in release builds.
//! - `WGPU_VALIDATION=0` disables validation layers. This can help with particularly spammy errors.
//! - `WGPU_FORCE_FALLBACK_ADAPTER=1` attempts to force software rendering. This typically matches what is used in CI.
//! - `WGPU_ADAPTER_NAME` allows selecting a specific adapter by name.
//! - `WGPU_SETTINGS_PRIO=webgl2` uses webgl2 limits.
//! - `WGPU_SETTINGS_PRIO=compatibility` uses webgpu limits.
//! - `VERBOSE_SHADER_ERROR=1` prints more detailed information about WGSL compilation errors, such as shader defs and shader entrypoint.

#![expect(missing_docs, reason = "Not all docs are written yet, see #3492.")]
#![expect(unsafe_code, reason = "Unsafe code is used to improve performance.")]
#![cfg_attr(
    any(docsrs, docsrs_dep),
    expect(
        internal_features,
        reason = "rustdoc_internals is needed for fake_variadic"
    )
)]
#![cfg_attr(any(docsrs, docsrs_dep), feature(doc_cfg, rustdoc_internals))]
#![doc(
    html_logo_url = "https://bevy.org/assets/icon.png",
    html_favicon_url = "https://bevy.org/assets/icon.png"
)]

#[cfg(target_pointer_width = "16")]
compile_error!("bevy_render cannot compile for a 16-bit platform.");

extern crate alloc;
extern crate core;

// Required to make proc macros work in bevy itself.
extern crate self as bevy_render;

pub mod batching;
pub mod camera;
pub mod diagnostic;
pub mod erased_render_asset;
pub mod error_handler;
pub mod extract_component;
pub mod globals;
pub mod gpu_component_array_buffer;
pub mod gpu_readback;
pub mod mesh;
pub mod occlusion_culling;
#[cfg(not(target_arch = "wasm32"))]
pub mod pipelined_rendering;
pub mod render_asset;
pub mod render_phase;
pub mod render_resource;
pub mod renderer;
pub mod settings;
pub mod storage;
pub mod texture;
pub mod view;

/// The render prelude.
///
/// This includes the most common types in this crate, re-exported for your convenience.
pub mod prelude {
    #[doc(hidden)]
    pub use crate::{
        camera::NormalizedRenderTargetExt as _, texture::ManualTextureViews, view::Msaa,
    };
}

use crate::{
    camera::CameraPlugin,
    error_handler::{RenderErrorHandler, RenderState},
    gpu_readback::GpuReadbackPlugin,
    mesh::{MeshRenderAssetPlugin, RenderMesh},
    render_asset::prepare_assets,
    render_resource::PipelineCache,
    renderer::{render_system, RenderAdapterInfo},
    settings::RenderCreation,
    storage::StoragePlugin,
    texture::TexturePlugin,
    view::{ViewPlugin, WindowRenderPlugin},
};
use alloc::sync::Arc;
use batching::gpu_preprocessing::BatchingPlugin;
use bevy_app::{App, AppLabel, Plugin};
use bevy_asset::{AssetApp, AssetServer};
use bevy_derive::Deref;
use bevy_ecs::{prelude::*, schedule::ScheduleLabel};
use bevy_extract::{ExtractPlugin, ExtractSchedule, Render, RenderApp, RenderSystems};
use bevy_platform::time::Instant;
use bevy_shader::{load_shader_library, Shader, ShaderLoader};
use bevy_time::TimeSender;
use bevy_window::{PrimaryWindow, RawHandleWrapperHolder};
use bitflags::bitflags;
use globals::GlobalsPlugin;
use occlusion_culling::OcclusionCullingPlugin;
use render_asset::{
    extract_render_asset_bytes_per_frame, reset_render_asset_bytes_per_frame,
    RenderAssetBytesPerFrame, RenderAssetBytesPerFrameLimiter,
};
use settings::RenderResources;
use std::sync::Mutex;

/// Contains the default Bevy rendering backend based on wgpu.
///
/// Rendering is done in a [`SubApp`](bevy_app::SubApp), which exchanges data with the main app
/// between main schedule iterations.
///
/// Rendering can be executed between iterations of the main schedule,
/// or it can be executed in parallel with main schedule when
/// [`PipelinedRenderingPlugin`](pipelined_rendering::PipelinedRenderingPlugin) is enabled.
#[derive(Default)]
pub struct RenderPlugin {
    pub render_creation: RenderCreation,
    /// If `true`, disables asynchronous pipeline compilation.
    /// This has no effect on macOS, Wasm, iOS, or without the `multi_threaded` feature.
    pub synchronous_pipeline_compilation: bool,
    /// Debugging flags that can optionally be set when constructing the renderer.
    pub debug_flags: RenderDebugFlags,
}

bitflags! {
    /// Debugging flags that can optionally be set when constructing the renderer.
    #[derive(Clone, Copy, PartialEq, Default, Debug)]
    pub struct RenderDebugFlags: u8 {
        /// If true, this sets the `COPY_SRC` flag on indirect draw parameters
        /// so that they can be read back to CPU.
        ///
        /// This is a debugging feature that may reduce performance. It
        /// primarily exists for the `occlusion_culling` example.
        const ALLOW_COPIES_FROM_INDIRECT_PARAMETERS = 1;
    }
}

// TODO: EXTRACT: maybe move these to bevy_extract along with e.g. Render schedule

/// The startup schedule of the [`RenderApp`].
/// This can potentially run multiple times, and not on a fresh render world.
/// Every time a new [`RenderDevice`](renderer::RenderDevice) is acquired,
/// this schedule runs to initialize any gpu resources needed for rendering on it.
#[derive(ScheduleLabel, Debug, Hash, PartialEq, Eq, Clone, Default)]
pub struct RenderStartup;

/// The render recovery schedule. This schedule runs the [`Render`] schedule if
/// we are in [`RenderState::Ready`], and is otherwise hidden from users.
#[derive(ScheduleLabel, Debug, Hash, PartialEq, Eq, Clone)]
struct RenderRecovery;

#[derive(Resource, Default, Clone, Deref)]
pub(crate) struct FutureRenderResources(Arc<Mutex<Option<RenderResources>>>);

impl Plugin for RenderPlugin {
    /// Initializes the renderer, sets up the [`RenderSystems`] and creates the rendering sub-app.
    fn build(&self, app: &mut App) {
        app.init_asset::<Shader>()
            .init_asset_loader::<ShaderLoader>();
        load_shader_library!(app, "maths.wgsl");
        load_shader_library!(app, "color_operations.wgsl");
        load_shader_library!(app, "bindless.wgsl");

        if insert_future_resources(&self.render_creation, app.world_mut()) {
            // We only create the render world and set up extraction if we
            // have a rendering backend available.
            app.add_plugins(ExtractPlugin {
                pre_extract: error_handler::update_state,
            });
        };

        app.add_plugins((
            WindowRenderPlugin,
            CameraPlugin,
            ViewPlugin,
            MeshRenderAssetPlugin,
            GlobalsPlugin,
            #[cfg(feature = "morph")]
            mesh::MorphPlugin,
            TexturePlugin,
            BatchingPlugin {
                debug_flags: self.debug_flags,
            },
            StoragePlugin,
            GpuReadbackPlugin::default(),
            OcclusionCullingPlugin,
            #[cfg(feature = "tracing-tracy")]
            diagnostic::RenderDiagnosticsPlugin,
        ));

        let asset_server = app.world().resource::<AssetServer>().clone();
        app.init_resource::<RenderAssetBytesPerFrame>()
            .init_resource::<RenderErrorHandler>();
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app.init_resource::<RenderAssetBytesPerFrameLimiter>();
            render_app.init_resource::<renderer::PendingCommandBuffers>();
            render_app.insert_resource(asset_server);
            render_app.insert_resource(RenderState::Initializing);
            render_app.add_systems(
                ExtractSchedule,
                (
                    extract_render_asset_bytes_per_frame,
                    PipelineCache::extract_shaders,
                ),
            );

            render_app.init_schedule(RenderStartup);
            render_app.update_schedule = Some(RenderRecovery.intern());
            render_app.add_systems(
                RenderRecovery,
                (run_render_schedule.run_if(renderer_is_ready), send_time).chain(),
            );
            render_app.add_systems(
                Render,
                (
                    (PipelineCache::process_pipeline_queue_system, render_system)
                        .chain()
                        .in_set(RenderSystems::Render),
                    reset_render_asset_bytes_per_frame.in_set(RenderSystems::Cleanup),
                ),
            );
        }
    }

    fn ready(&self, app: &App) -> bool {
        // This is a little tricky. `FutureRenderResources` is added in `build`, which runs synchronously before `ready`.
        // It is only added if there is a wgpu backend and thus the renderer can be created.
        // Hence, if we try and get the resource and it is not present, that means we are ready, because we dont need it.
        // On the other hand, if the resource is present, then we try and lock on it. The lock can fail, in which case
        // we currently can assume that means the `FutureRenderResources` is in the act of being populated, because
        // that is the only other place the lock may be held. If it is being populated, we can assume we're ready. This
        // happens via the `and_then` falling through to the same `unwrap_or(true)` case as when there's no resource.
        // If the lock succeeds, we can straightforwardly check if it is populated. If it is not, then we're not ready.
        app.world()
            .get_resource::<FutureRenderResources>()
            .and_then(|frr| frr.try_lock().map(|locked| locked.is_some()).ok())
            .unwrap_or(true)
    }

    fn finish(&self, app: &mut App) {
        if let Some(future_render_resources) =
            app.world_mut().remove_resource::<FutureRenderResources>()
        {
            let bevy_app::SubApps { main, sub_apps } = app.sub_apps_mut();
            let render = sub_apps.get_mut(&RenderApp.intern()).unwrap();
            let render_resources = future_render_resources.0.lock().unwrap().take().unwrap();

            render_resources.unpack_into(
                main.world_mut(),
                render.world_mut(),
                self.synchronous_pipeline_compilation,
            );
        }
    }
}

fn renderer_is_ready(state: Res<RenderState>) -> bool {
    matches!(*state, RenderState::Ready)
}

fn run_render_schedule(world: &mut World) {
    world.run_schedule(Render);
}

fn send_time(time_sender: Res<TimeSender>) {
    // update the time and send it to the app world regardless of whether we render
    if let Err(error) = time_sender.0.try_send(Instant::now()) {
        match error {
            bevy_time::TrySendError::Full(_) => {
                panic!(
                    "The TimeSender channel should always be empty during render. \
                            You might need to add the bevy::core::time_system to your app."
                );
            }
            bevy_time::TrySendError::Disconnected(_) => {
                // ignore disconnected errors, the main world probably just got dropped during shutdown
            }
        }
    }
}

/// Inserts a [`FutureRenderResources`] created from this [`RenderCreation`].
///
/// Returns true if creation was successful, false otherwise.
fn insert_future_resources(render_creation: &RenderCreation, main_world: &mut World) -> bool {
    let primary_window = main_world
        .query_filtered::<&RawHandleWrapperHolder, With<PrimaryWindow>>()
        .single(main_world)
        .ok()
        .cloned();

    #[cfg(feature = "raw_vulkan_init")]
    let raw_vulkan_init_settings = main_world
        .get_resource::<renderer::raw_vulkan_init::RawVulkanInitSettings>()
        .cloned()
        .unwrap_or_default();

    let future_resources = FutureRenderResources::default();
    let success = render_creation.create_render(
        future_resources.clone(),
        primary_window,
        #[cfg(feature = "raw_vulkan_init")]
        raw_vulkan_init_settings,
    );
    if success {
        // Note that `future_resources` is not necessarily populated here yet.
        main_world.insert_resource(future_resources);
    }
    success
}

/// If the [`RenderAdapterInfo`] is a Qualcomm Adreno, returns its model number.
///
/// This lets us work around hardware bugs.
pub fn get_adreno_model(adapter_info: &RenderAdapterInfo) -> Option<u32> {
    if !cfg!(target_os = "android") {
        return None;
    }

    let adreno_model = adapter_info.name.strip_prefix("Adreno (TM) ")?;

    // Take suffixes into account (like Adreno 642L).
    Some(
        adreno_model
            .chars()
            .map_while(|c| c.to_digit(10))
            .fold(0, |acc, digit| acc * 10 + digit),
    )
}

/// Get the Mali driver version if the adapter is a Mali GPU.
pub fn get_mali_driver_version(adapter_info: &RenderAdapterInfo) -> Option<u32> {
    if !cfg!(target_os = "android") {
        return None;
    }

    if !adapter_info.name.contains("Mali") {
        return None;
    }
    let driver_info = &adapter_info.driver_info;
    if let Some(start_pos) = driver_info.find("v1.r")
        && let Some(end_pos) = driver_info[start_pos..].find('p')
    {
        let start_idx = start_pos + 4; // Skip "v1.r"
        let end_idx = start_pos + end_pos;

        return driver_info[start_idx..end_idx].parse::<u32>().ok();
    }

    None
}
