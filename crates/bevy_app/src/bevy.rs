use crate::{
    App, Apps, First, Main, MainSchedulePlugin, PlaceholderPlugin, Plugin, Plugins, PluginsState,
};
use alloc::{
    boxed::Box,
    string::{String, ToString},
};
pub use bevy_derive::AppLabel;
use bevy_ecs::{
    component::RequiredComponentsError,
    error::{ErrorHandler, FallbackErrorHandler},
    intern::Interned,
    message::{message_update_system, MessageCursor},
    prelude::*,
    schedule::ScheduleLabel,
};
use bevy_platform::collections::HashMap;
#[cfg(feature = "bevy_reflect")]
use bevy_reflect::Reflect;
use core::{fmt::Debug, num::NonZero, panic::AssertUnwindSafe};
use std::println;
use log::debug;

#[cfg(feature = "trace")]
use tracing::info_span;

#[cfg(feature = "std")]
use std::{
    panic::{catch_unwind, resume_unwind},
    process::{ExitCode, Termination},
};

bevy_ecs::define_label!(
    /// A strongly-typed class of labels used to identify an [`App`].
    #[diagnostic::on_unimplemented(
        note = "consider annotating `{Self}` with `#[derive(AppLabel)]`"
    )]
    AppLabel,
);

pub use bevy_ecs::label::DynEq;

/// A shorthand for `Interned<dyn AppLabel>`.
pub type InternedAppLabel = Interned<dyn AppLabel>;

#[derive(Debug, thiserror::Error)]
pub(crate) enum AppError {
    #[error("duplicate plugin {plugin_name:?}")]
    DuplicatePlugin { plugin_name: String },
}

/// [`App`] is the primary API for writing user applications. It automates the setup of a
/// [standard lifecycle](Main) and provides interface glue for [plugins](`Plugin`).
///
/// A single [`App`] can contain multiple [`App`] instances, but [`App`] methods only affect
/// the "main" one. To access a particular [`App`], use [`get_app`](App::get_app)
/// or [`get_app_mut`](App::get_app_mut).
///
///
/// # Examples
///
/// Here is a simple "Hello World" Bevy app:
///
/// ```
/// # use bevy_app::prelude::*;
/// # use bevy_ecs::prelude::*;
/// #
/// fn main() {
///    App::new()
///        .add_systems(Update, hello_world_system)
///        .run();
/// }
///
/// fn hello_world_system() {
///    println!("hello world");
/// }
/// ```
#[must_use]
pub struct Bevy {
    pub(crate) apps: Apps,
    /// The function that will manage the app's lifecycle.
    ///
    /// Bevy provides the [`WinitPlugin`] and [`ScheduleRunnerPlugin`] for windowed and headless
    /// applications, respectively.
    ///
    /// [`WinitPlugin`]: https://docs.rs/bevy/latest/bevy/winit/struct.WinitPlugin.html
    /// [`ScheduleRunnerPlugin`]: https://docs.rs/bevy/latest/bevy/app/struct.ScheduleRunnerPlugin.html
    pub(crate) runner: RunnerFn,
    fallback_error_handler: Option<ErrorHandler>,
}

impl Debug for Bevy {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "App {{ apps: ")?;
        f.debug_map().entries(self.apps.apps.iter()).finish()?;
        write!(f, "}}")
    }
}

impl Default for Bevy {
    fn default() -> Self {
        let mut bevy = Bevy::empty();
        bevy.apps.main.update_schedule = Some(Main.intern());

        let app = bevy.main_mut();

        #[cfg(feature = "bevy_reflect")]
        {
            #[cfg(not(feature = "reflect_auto_register"))]
            app.init_resource::<AppTypeRegistry>();

            #[cfg(feature = "reflect_auto_register")]
            app.insert_resource(AppTypeRegistry::new_with_derived_types());
        }

        #[cfg(feature = "reflect_functions")]
        app.init_resource::<AppFunctionRegistry>();

        app.add_plugins(MainSchedulePlugin);
        app.add_systems(
            First,
            message_update_system
                .in_set(bevy_ecs::message::MessageUpdateSystems)
                .run_if(bevy_ecs::message::message_update_condition),
        );
        app.add_systems(
            crate::Last,
            bevy_ecs::system::despawn_unused_registered_systems,
        );
        app.add_message::<AppExit>();

        bevy
    }
}

impl Bevy {
    /// Creates a new [`App`] with some default structure to enable core engine features.
    /// This is the preferred constructor for most use cases.
    pub fn new() -> Bevy {
        Bevy::default()
    }

    /// Creates a new empty [`App`] with minimal default configuration.
    ///
    /// Use this constructor if you want to customize scheduling, exit handling, cleanup, etc.
    pub fn empty() -> Bevy {
        Self {
            apps: Apps {
                main: App::new(),
                apps: HashMap::default(),
            },
            runner: Box::new(run_once),
            fallback_error_handler: None,
        }
    }

    /// Runs the default schedules of all sub-apps (starting with the "main" app) once.
    pub fn update(&mut self) {
        if self.is_building_plugins() {
            panic!("App::update() was called while a plugin was building.");
        }

        self.apps.update();
    }

    /// Runs the [`App`] by calling its [runner](Self::set_runner).
    ///
    /// This will (re)build the [`App`] first. For general usage, see the example on the item
    /// level documentation.
    ///
    /// # Caveats
    ///
    /// Calls to [`App::run()`] will never return on iOS and Web.
    ///
    /// Headless apps can generally expect this method to return control to the caller when
    /// it completes, but that is not the case for windowed apps. Windowed apps are typically
    /// driven by an event loop and some platforms expect the program to terminate when the
    /// event loop ends.
    ///
    /// By default, *Bevy* uses the `winit` crate for window creation.
    ///
    /// # Panics
    ///
    /// Panics if not all plugins have been built.
    pub fn run(&mut self) -> AppExit {
        #[cfg(feature = "trace")]
        let _bevy_app_run_span = info_span!("bevy_app").entered();
        if self.is_building_plugins() {
            panic!("App::run() was called while a plugin was building.");
        }

        let runner = core::mem::replace(&mut self.runner, Box::new(run_once));
        let app = core::mem::replace(self, Bevy::empty());
        (runner)(app)
    }

    /// Sets the function that will be called when the app is run.
    ///
    /// The runner function `f` is called only once by [`App::run`]. If the
    /// presence of a main loop in the app is desired, it is the responsibility of the runner
    /// function to provide it.
    ///
    /// The runner function is usually not set manually, but by Bevy integrated plugins
    /// (e.g. `WinitPlugin`).
    ///
    /// # Examples
    ///
    /// ```
    /// # use bevy_app::prelude::*;
    /// #
    /// fn my_runner(mut app: App) -> AppExit {
    ///     loop {
    ///         println!("In main loop");
    ///         app.update();
    ///         if let Some(exit) = app.should_exit() {
    ///             return exit;
    ///         }
    ///     }
    /// }
    ///
    /// App::new()
    ///     .set_runner(my_runner);
    /// ```
    pub fn set_runner(&mut self, f: impl FnOnce(Bevy) -> AppExit + 'static) -> &mut Self {
        self.runner = Box::new(f);
        self
    }

    /// Returns the state of all plugins. This is usually called by the event loop, but can be
    /// useful for situations where you want to use [`App::update`].
    // TODO: &mut self -> &self
    #[inline]
    pub fn plugins_state(&mut self) -> PluginsState {
        let mut overall_plugins_state = match self.main_mut().plugins_state {
            PluginsState::Adding => {
                let mut state = PluginsState::Ready;
                let plugins = core::mem::take(&mut self.main_mut().plugin_registry);
                for plugin in &plugins {
                    // plugins installed to main need to see all sub-apps
                    if !plugin.ready(self) {
                        state = PluginsState::Adding;
                        break;
                    }
                }
                self.main_mut().plugin_registry = plugins;
                state
            }
            state => state,
        };

        // overall state is the earliest state of any sub-app
        self.apps.iter_mut().skip(1).for_each(|s| {
            overall_plugins_state = overall_plugins_state.min(s.plugins_state());
        });

        overall_plugins_state
    }

    /// Runs [`Plugin::finish`] for each plugin. This is usually called by the event loop once all
    /// plugins are ready, but can be useful for situations where you want to use [`App::update`].
    pub fn finish(&mut self) {
        #[cfg(feature = "trace")]
        let _finish_span = info_span!("plugin finish").entered();
        // plugins installed to main should see all sub-apps
        // do hokey pokey with a boxed zst plugin (doesn't allocate)
        let mut hokeypokey: Box<dyn Plugin> = Box::new(HokeyPokey);
        for i in 0..self.main().plugin_registry.len() {
            core::mem::swap(&mut self.main_mut().plugin_registry[i], &mut hokeypokey);
            #[cfg(feature = "trace")]
            let _plugin_finish_span =
                info_span!("plugin finish", plugin = hokeypokey.name()).entered();
            hokeypokey.finish(self);
            core::mem::swap(&mut self.main_mut().plugin_registry[i], &mut hokeypokey);
        }
        self.main_mut().plugins_state = PluginsState::Finished;
        self.apps.iter_mut().skip(1).for_each(App::finish);
    }

    /// Runs [`Plugin::cleanup`] for each plugin. This is usually called by the event loop after
    /// [`App::finish`], but can be useful for situations where you want to use [`App::update`].
    pub fn cleanup(&mut self) {
        #[cfg(feature = "trace")]
        let _cleanup_span = info_span!("plugin cleanup").entered();
        // plugins installed to main should see all sub-apps
        // do hokey pokey with a boxed zst plugin (doesn't allocate)
        let mut hokeypokey: Box<dyn Plugin> = Box::new(HokeyPokey);
        for i in 0..self.main().plugin_registry.len() {
            core::mem::swap(&mut self.main_mut().plugin_registry[i], &mut hokeypokey);
            #[cfg(feature = "trace")]
            let _plugin_cleanup_span =
                info_span!("plugin cleanup", plugin = hokeypokey.name()).entered();
            hokeypokey.cleanup(self);
            core::mem::swap(&mut self.main_mut().plugin_registry[i], &mut hokeypokey);
        }
        self.main_mut().plugins_state = PluginsState::Cleaned;
        self.apps.iter_mut().skip(1).for_each(App::cleanup);
    }

    /// Returns `true` if any of the sub-apps are building plugins.
    pub(crate) fn is_building_plugins(&self) -> bool {
        self.apps.iter().any(App::is_building_plugins)
    }

    /// Inserts the [`!Send`](Send) resource into the app, overwriting any existing data
    /// of the same type.
    #[deprecated(since = "0.19.0", note = "use App::insert_non_send")]
    pub fn insert_non_send_resource<R: 'static>(&mut self, resource: R) -> &mut Self {
        self.insert_non_send(resource)
    }

    /// Inserts the [`!Send`](Send) data into the app, overwriting any existing data
    /// of the same type.
    ///
    /// There is also an [`init_non_send`](Self::init_non_send) for [`!Send`](Send) data
    /// that implement [`Default`]
    ///
    /// # Examples
    ///
    /// ```
    /// # use bevy_app::prelude::*;
    /// # use bevy_ecs::prelude::*;
    /// #
    /// struct MyCounter {
    ///     counter: usize,
    /// }
    ///
    /// App::new()
    ///     .insert_non_send(MyCounter { counter: 0 });
    /// ```
    pub fn insert_non_send<R: 'static>(&mut self, resource: R) -> &mut Self {
        self.world_mut().insert_non_send(resource);
        self
    }

    /// Inserts the [`!Send`](Send) resource into the app if there is no existing instance of `R`.
    #[deprecated(since = "0.19.0", note = "use App::init_non_send")]
    pub fn init_non_send_resource<R: 'static + FromWorld>(&mut self) -> &mut Self {
        self.init_non_send::<R>()
    }

    /// Inserts the [`!Send`](Send) data into the app if there is no existing instance of `R`.
    ///
    /// `R` must implement [`FromWorld`].
    /// If `R` implements [`Default`], [`FromWorld`] will be automatically implemented and
    /// initialize the [`Resource`] with [`Default::default`].
    pub fn init_non_send<R: 'static + FromWorld>(&mut self) -> &mut Self {
        self.world_mut().init_non_send::<R>();
        self
    }

    pub(crate) fn add_boxed_plugin(
        &mut self,
        plugin: Box<dyn Plugin>,
    ) -> Result<&mut Self, AppError> {
        debug!("added plugin: {}", plugin.name());
        if plugin.is_unique() && self.main_mut().plugin_names.contains(plugin.name()) {
            Err(AppError::DuplicatePlugin {
                plugin_name: plugin.name().to_string(),
            })?;
        }

        // Reserve position in the plugin registry. If the plugin adds more plugins,
        // they'll all end up in insertion order.
        let index = self.main().plugin_registry.len();
        self.main_mut()
            .plugin_registry
            .push(Box::new(PlaceholderPlugin));

        self.main_mut().plugin_build_depth += 1;

        #[cfg(feature = "trace")]
        let _plugin_build_span = info_span!("plugin build", plugin = plugin.name()).entered();

        let f = AssertUnwindSafe(|| plugin.build(self));

        #[cfg(feature = "std")]
        let result = catch_unwind(f);

        #[cfg(not(feature = "std"))]
        f();

        self.main_mut()
            .plugin_names
            .insert(plugin.name().to_string());
        self.main_mut().plugin_build_depth -= 1;

        #[cfg(feature = "std")]
        if let Err(payload) = result {
            resume_unwind(payload);
        }

        self.main_mut().plugin_registry[index] = plugin;
        Ok(self)
    }

    /// Installs a [`Plugin`] collection.
    ///
    /// Bevy prioritizes modularity as a core principle. **All** engine features are implemented
    /// as plugins, even the complex ones like rendering.
    ///
    /// [`Plugin`]s can be grouped into a set by using a [`PluginGroup`].
    ///
    /// There are built-in [`PluginGroup`]s that provide core engine functionality.
    /// The [`PluginGroup`]s available by default are `DefaultPlugins` and `MinimalPlugins`.
    ///
    /// To customize the plugins in the group (reorder, disable a plugin, add a new plugin
    /// before / after another plugin), call [`build()`](super::PluginGroup::build) on the group,
    /// which will convert it to a [`PluginGroupBuilder`](crate::PluginGroupBuilder).
    ///
    /// You can also specify a group of [`Plugin`]s by using a tuple over [`Plugin`]s and
    /// [`PluginGroup`]s. See [`Plugins`] for more details.
    ///
    /// ## Examples
    /// ```
    /// # use bevy_app::{prelude::*, PluginGroupBuilder, NoopPluginGroup as MinimalPlugins};
    /// #
    /// # // Dummies created to avoid using `bevy_log`,
    /// # // which pulls in too many dependencies and breaks rust-analyzer
    /// # pub struct LogPlugin;
    /// # impl Plugin for LogPlugin {
    /// #     fn build(&self, app: &mut App) {}
    /// # }
    /// App::new()
    ///     .add_plugins(MinimalPlugins);
    /// App::new()
    ///     .add_plugins((MinimalPlugins, LogPlugin));
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if one of the plugins had already been added to the application.
    ///
    /// [`PluginGroup`]:super::PluginGroup
    #[track_caller]
    pub fn add_plugins<M>(&mut self, plugins: impl Plugins<M>) -> &mut Self {
        if matches!(
            self.plugins_state(),
            PluginsState::Cleaned | PluginsState::Finished
        ) {
            panic!(
                "Plugins cannot be added after App::cleanup() or App::finish() has been called."
            );
        }
        plugins.add_to_bevy(self);
        self
    }

    /// Registers the given component `R` as a [required component] for `T`.
    ///
    /// When `T` is added to an entity, `R` and its own required components will also be added
    /// if `R` was not already provided. The [`Default`] `constructor` will be used for the creation of `R`.
    /// If a custom constructor is desired, use [`App::register_required_components_with`] instead.
    ///
    /// For the non-panicking version, see [`App::try_register_required_components`].
    ///
    /// Note that requirements must currently be registered before `T` is inserted into the world
    /// for the first time. Commonly, this is done in plugins. This limitation may be fixed in the future.
    ///
    /// [required component]: Component#required-components
    ///
    /// # Panics
    ///
    /// Panics if `R` is already a directly required component for `T`, or if `T` has ever been added
    /// on an entity before the registration.
    ///
    /// Indirect requirements through other components are allowed. In those cases, any existing requirements
    /// will only be overwritten if the new requirement is more specific.
    ///
    /// # Example
    ///
    /// ```
    /// # use bevy_app::{App, NoopPluginGroup as MinimalPlugins, Startup};
    /// # use bevy_ecs::prelude::*;
    /// #[derive(Component)]
    /// struct A;
    ///
    /// #[derive(Component, Default, PartialEq, Eq, Debug)]
    /// struct B(usize);
    ///
    /// #[derive(Component, Default, PartialEq, Eq, Debug)]
    /// struct C(u32);
    ///
    /// # let mut app = App::new();
    /// # app.add_plugins(MinimalPlugins).add_systems(Startup, setup);
    /// // Register B as required by A and C as required by B.
    /// app.register_required_components::<A, B>();
    /// app.register_required_components::<B, C>();
    ///
    /// fn setup(mut commands: Commands) {
    ///     // This will implicitly also insert B and C with their Default constructors.
    ///     commands.spawn(A);
    /// }
    ///
    /// fn validate(query: Option<Single<(&A, &B, &C)>>) {
    ///     let (a, b, c) = query.unwrap().into_inner();
    ///     assert_eq!(b, &B(0));
    ///     assert_eq!(c, &C(0));
    /// }
    /// # app.update();
    /// ```
    pub fn register_required_components<T: Component, R: Component + Default>(
        &mut self,
    ) -> &mut Self {
        self.world_mut().register_required_components::<T, R>();
        self
    }

    /// Registers the given component `R` as a [required component] for `T`.
    ///
    /// When `T` is added to an entity, `R` and its own required components will also be added
    /// if `R` was not already provided. The given `constructor` will be used for the creation of `R`.
    /// If a [`Default`] constructor is desired, use [`App::register_required_components`] instead.
    ///
    /// For the non-panicking version, see [`App::try_register_required_components_with`].
    ///
    /// Note that requirements must currently be registered before `T` is inserted into the world
    /// for the first time. Commonly, this is done in plugins. This limitation may be fixed in the future.
    ///
    /// [required component]: Component#required-components
    ///
    /// # Panics
    ///
    /// Panics if `R` is already a directly required component for `T`, or if `T` has ever been added
    /// on an entity before the registration.
    ///
    /// Indirect requirements through other components are allowed. In those cases, any existing requirements
    /// will only be overwritten if the new requirement is more specific.
    ///
    /// # Example
    ///
    /// ```
    /// # use bevy_app::{App, NoopPluginGroup as MinimalPlugins, Startup};
    /// # use bevy_ecs::prelude::*;
    /// #[derive(Component)]
    /// struct A;
    ///
    /// #[derive(Component, Default, PartialEq, Eq, Debug)]
    /// struct B(usize);
    ///
    /// #[derive(Component, Default, PartialEq, Eq, Debug)]
    /// struct C(u32);
    ///
    /// # let mut app = App::new();
    /// # app.add_plugins(MinimalPlugins).add_systems(Startup, setup);
    /// // Register B and C as required by A and C as required by B.
    /// // A requiring C directly will overwrite the indirect requirement through B.
    /// app.register_required_components::<A, B>();
    /// app.register_required_components_with::<B, C>(|| C(1));
    /// app.register_required_components_with::<A, C>(|| C(2));
    ///
    /// fn setup(mut commands: Commands) {
    ///     // This will implicitly also insert B with its Default constructor and C
    ///     // with the custom constructor defined by A.
    ///     commands.spawn(A);
    /// }
    ///
    /// fn validate(query: Option<Single<(&A, &B, &C)>>) {
    ///     let (a, b, c) = query.unwrap().into_inner();
    ///     assert_eq!(b, &B(0));
    ///     assert_eq!(c, &C(2));
    /// }
    /// # app.update();
    /// ```
    pub fn register_required_components_with<T: Component, R: Component>(
        &mut self,
        constructor: fn() -> R,
    ) -> &mut Self {
        self.world_mut()
            .register_required_components_with::<T, R>(constructor);
        self
    }

    /// Tries to register the given component `R` as a [required component] for `T`.
    ///
    /// When `T` is added to an entity, `R` and its own required components will also be added
    /// if `R` was not already provided. The [`Default`] `constructor` will be used for the creation of `R`.
    /// If a custom constructor is desired, use [`App::register_required_components_with`] instead.
    ///
    /// For the panicking version, see [`App::register_required_components`].
    ///
    /// Note that requirements must currently be registered before `T` is inserted into the world
    /// for the first time. Commonly, this is done in plugins. This limitation may be fixed in the future.
    ///
    /// [required component]: Component#required-components
    ///
    /// # Errors
    ///
    /// Returns a [`RequiredComponentsError`] if `R` is already a directly required component for `T`, or if `T` has ever been added
    /// on an entity before the registration.
    ///
    /// Indirect requirements through other components are allowed. In those cases, any existing requirements
    /// will only be overwritten if the new requirement is more specific.
    ///
    /// # Example
    ///
    /// ```
    /// # use bevy_app::{App, NoopPluginGroup as MinimalPlugins, Startup};
    /// # use bevy_ecs::prelude::*;
    /// #[derive(Component)]
    /// struct A;
    ///
    /// #[derive(Component, Default, PartialEq, Eq, Debug)]
    /// struct B(usize);
    ///
    /// #[derive(Component, Default, PartialEq, Eq, Debug)]
    /// struct C(u32);
    ///
    /// # let mut app = App::new();
    /// # app.add_plugins(MinimalPlugins).add_systems(Startup, setup);
    /// // Register B as required by A and C as required by B.
    /// app.register_required_components::<A, B>();
    /// app.register_required_components::<B, C>();
    ///
    /// // Duplicate registration! This will fail.
    /// assert!(app.try_register_required_components::<A, B>().is_err());
    ///
    /// fn setup(mut commands: Commands) {
    ///     // This will implicitly also insert B and C with their Default constructors.
    ///     commands.spawn(A);
    /// }
    ///
    /// fn validate(query: Option<Single<(&A, &B, &C)>>) {
    ///     let (a, b, c) = query.unwrap().into_inner();
    ///     assert_eq!(b, &B(0));
    ///     assert_eq!(c, &C(0));
    /// }
    /// # app.update();
    /// ```
    pub fn try_register_required_components<T: Component, R: Component + Default>(
        &mut self,
    ) -> Result<(), RequiredComponentsError> {
        self.world_mut().try_register_required_components::<T, R>()
    }

    /// Tries to register the given component `R` as a [required component] for `T`.
    ///
    /// When `T` is added to an entity, `R` and its own required components will also be added
    /// if `R` was not already provided. The given `constructor` will be used for the creation of `R`.
    /// If a [`Default`] constructor is desired, use [`App::register_required_components`] instead.
    ///
    /// For the panicking version, see [`App::register_required_components_with`].
    ///
    /// Note that requirements must currently be registered before `T` is inserted into the world
    /// for the first time. Commonly, this is done in plugins. This limitation may be fixed in the future.
    ///
    /// [required component]: Component#required-components
    ///
    /// # Errors
    ///
    /// Returns a [`RequiredComponentsError`] if `R` is already a directly required component for `T`, or if `T` has ever been added
    /// on an entity before the registration.
    ///
    /// Indirect requirements through other components are allowed. In those cases, any existing requirements
    /// will only be overwritten if the new requirement is more specific.
    ///
    /// # Example
    ///
    /// ```
    /// # use bevy_app::{App, NoopPluginGroup as MinimalPlugins, Startup};
    /// # use bevy_ecs::prelude::*;
    /// #[derive(Component)]
    /// struct A;
    ///
    /// #[derive(Component, Default, PartialEq, Eq, Debug)]
    /// struct B(usize);
    ///
    /// #[derive(Component, Default, PartialEq, Eq, Debug)]
    /// struct C(u32);
    ///
    /// # let mut app = App::new();
    /// # app.add_plugins(MinimalPlugins).add_systems(Startup, setup);
    /// // Register B and C as required by A and C as required by B.
    /// // A requiring C directly will overwrite the indirect requirement through B.
    /// app.register_required_components::<A, B>();
    /// app.register_required_components_with::<B, C>(|| C(1));
    /// app.register_required_components_with::<A, C>(|| C(2));
    ///
    /// // Duplicate registration! Even if the constructors were different, this would fail.
    /// assert!(app.try_register_required_components_with::<B, C>(|| C(1)).is_err());
    ///
    /// fn setup(mut commands: Commands) {
    ///     // This will implicitly also insert B with its Default constructor and C
    ///     // with the custom constructor defined by A.
    ///     commands.spawn(A);
    /// }
    ///
    /// fn validate(query: Option<Single<(&A, &B, &C)>>) {
    ///     let (a, b, c) = query.unwrap().into_inner();
    ///     assert_eq!(b, &B(0));
    ///     assert_eq!(c, &C(2));
    /// }
    /// # app.update();
    /// ```
    pub fn try_register_required_components_with<T: Component, R: Component>(
        &mut self,
        constructor: fn() -> R,
    ) -> Result<(), RequiredComponentsError> {
        self.world_mut()
            .try_register_required_components_with::<T, R>(constructor)
    }

    /// Registers a component type as "disabling",
    /// using [default query filters](bevy_ecs::entity_disabling::DefaultQueryFilters) to exclude entities with the component from queries.
    ///
    /// # Warning
    ///
    /// As discussed in the [module docs](bevy_ecs::entity_disabling), this can have performance implications,
    /// as well as create interoperability issues, and should be used with caution.
    pub fn register_disabling_component<C: Component>(&mut self) {
        self.world_mut().register_disabling_component::<C>();
    }

    /// Returns a reference to the main [`App`]'s [`World`]. This is the same as calling
    /// [`app.main().world()`].
    ///
    /// [`app.main().world()`]: App::world
    pub fn world(&self) -> &World {
        self.main().world()
    }

    /// Returns a mutable reference to the main [`App`]'s [`World`]. This is the same as calling
    /// [`app.main_mut().world_mut()`].
    ///
    /// [`app.main_mut().world_mut()`]: App::world_mut
    pub fn world_mut(&mut self) -> &mut World {
        self.main_mut().world_mut()
    }

    /// Returns a reference to the main [`App`].
    pub fn main(&self) -> &App {
        // println!("main");
        &self.apps.main
    }

    /// Returns a mutable reference to the main [`App`].
    pub fn main_mut(&mut self) -> &mut App {
        // println!("main_mut");
        &mut self.apps.main
    }

    /// Returns a reference to the [`Apps`] collection.
    pub fn apps(&self) -> &Apps {
        println!("apps");
        &self.apps
    }

    /// Returns a mutable reference to the [`Apps`] collection.
    pub fn apps_mut(&mut self) -> &mut Apps {
        println!("apps_mut");
        &mut self.apps
    }

    /// Returns a reference to the [`App`] with the given label.
    ///
    /// # Panics
    ///
    /// Panics if the [`App`] doesn't exist.
    pub fn app(&self, label: impl AppLabel) -> &App {
        println!("app {:?}", label);
        let str = label.intern();
        self.get_app(label).unwrap_or_else(|| {
            panic!("No sub-app with label '{:?}' exists.", str);
        })
    }

    /// Returns a reference to the [`App`] with the given label.
    ///
    /// # Panics
    ///
    /// Panics if the [`App`] doesn't exist.
    pub fn app_mut(&mut self, label: impl AppLabel) -> &mut App {
        println!("app_mut {:?}", label);
        let str = label.intern();
        self.get_app_mut(label).unwrap_or_else(|| {
            panic!("No sub-app with label '{:?}' exists.", str);
        })
    }

    /// Returns a reference to the [`App`] with the given label, if it exists.
    pub fn get_app(&self, label: impl AppLabel) -> Option<&App> {
        println!("get_app {:?}", label);
        self.apps.apps.get(&label.intern())
    }

    /// Returns a mutable reference to the [`App`] with the given label, if it exists.
    pub fn get_app_mut(&mut self, label: impl AppLabel) -> Option<&mut App> {
        println!("get_app_mut {:?}", label);
        println!("apps = {:?}", self.apps.apps);
        self.apps.apps.get_mut(&label.intern())
    }

    ///
    pub fn get_main_and_app_mut(&mut self, label: impl AppLabel) -> (&mut App, Option<&mut App>) {
        println!("get_main_and_app_mut {:?}", label);
        let Apps { main: app, apps } = self.apps_mut();
        println!("apps = {:?}", apps);
        let maybe_app = apps.get_mut(&label.intern());

        (app, maybe_app)
    }

    /// Inserts a [`App`] with the given label.
    pub fn insert_app(&mut self, label: impl AppLabel, mut app: App) {
        println!("insert_app {:?}", label);
        if let Some(handler) = self.fallback_error_handler {
            app.world_mut()
                .get_resource_or_insert_with(|| FallbackErrorHandler(handler));
        }
        self.apps.apps.insert(label.intern(), app);
        println!("apps = {:?}", self.apps.apps);
    }

    /// Removes the [`App`] with the given label, if it exists.
    pub fn remove_app(&mut self, label: impl AppLabel) -> Option<App> {
        println!("remove_app {:?}", label);
        self.apps.apps.remove(&label.intern())
    }

    /// Extract data from the main world into the [`App`] with the given label and perform an update if it exists.
    pub fn update_app_by_label(&mut self, label: impl AppLabel) {
        self.apps.update_app_by_label(label);
    }

    /// Attempts to determine if an [`AppExit`] was raised since the last update.
    ///
    /// Will attempt to return the first [`Error`](AppExit::Error) it encounters.
    /// This should be called after every [`update()`](App::update) otherwise you risk
    /// dropping possible [`AppExit`] events.
    pub fn should_exit(&self) -> Option<AppExit> {
        let mut reader = MessageCursor::default();

        let messages = self.world().get_resource::<Messages<AppExit>>()?;
        let mut messages = reader.read(messages);

        if messages.len() != 0 {
            return Some(
                messages
                    .find(|exit| exit.is_error())
                    .cloned()
                    .unwrap_or(AppExit::Success),
            );
        }

        None
    }

    /// Gets the error handler to set for new supapps.
    ///
    /// Note that the error handler of existing apps may differ.
    pub fn get_error_handler(&self) -> Option<ErrorHandler> {
        self.fallback_error_handler
    }

    /// Set the [fallback error handler] for the all apps (including the main one and future ones)
    /// that do not have one.
    ///
    /// May only be called once and should be set by the application, not by libraries.
    ///
    /// The handler will be called when an error is produced and not otherwise handled.
    ///
    /// # Panics
    /// Panics if called multiple times.
    ///
    /// # Example
    /// ```
    /// # use bevy_app::*;
    /// # use bevy_ecs::error::warn;
    /// # fn MyPlugins(_: &mut App) {}
    /// App::new()
    ///     .set_error_handler(warn)
    ///     .add_plugins(MyPlugins)
    ///     .run();
    /// ```
    ///
    /// [fallback error handler]: bevy_ecs::error::FallbackErrorHandler
    pub fn set_error_handler(&mut self, handler: ErrorHandler) -> &mut Self {
        assert!(
            self.fallback_error_handler.is_none(),
            "`set_error_handler` called multiple times on same `App`"
        );
        self.fallback_error_handler = Some(handler);
        for app in self.apps.iter_mut() {
            app.world_mut()
                .get_resource_or_insert_with(|| FallbackErrorHandler(handler));
        }
        self
    }
}

// Used for doing hokey pokey in finish and cleanup
pub(crate) struct HokeyPokey;
impl Plugin for HokeyPokey {
    fn build(&self, _: &mut Bevy) {}
}

type RunnerFn = Box<dyn FnOnce(Bevy) -> AppExit>;

fn run_once(mut app: Bevy) -> AppExit {
    while app.plugins_state() == PluginsState::Adding {
        #[cfg(not(all(target_arch = "wasm32", feature = "web")))]
        bevy_tasks::tick_global_task_pools_on_main_thread();
    }
    app.finish();
    app.cleanup();

    app.update();

    app.should_exit().unwrap_or(AppExit::Success)
}

/// A [`SystemSet`] for systems that should run before app exit (but
/// after an [`AppExit`] message has been sent).
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct OnAppExitSystems;

/// A [`Message`] that indicates the [`App`] should exit. If one or more of these are present at the end of an update,
/// the [runner](App::set_runner) will end and ([maybe](App::run)) return control to the caller.
///
/// This message can be used to detect when an exit is requested. Make sure that systems listening
/// for this message run before the current update ends.
///
/// # Portability
/// This type is roughly meant to map to a standard definition of a process exit code (0 means success, not 0 means error). Due to portability concerns
/// (see [`ExitCode`](https://doc.rust-lang.org/std/process/struct.ExitCode.html) and [`process::exit`](https://doc.rust-lang.org/std/process/fn.exit.html#))
/// we only allow error codes between 1 and [255](u8::MAX).
#[derive(Message, Debug, Clone, Default, PartialEq, Eq)]
#[cfg_attr(
    feature = "bevy_reflect",
    derive(Reflect),
    reflect(Debug, PartialEq, Clone, Message)
)]
pub enum AppExit {
    /// [`App`] exited without any problems.
    #[default]
    Success,
    /// The [`App`] experienced an unhandleable error.
    /// Holds the exit code we expect our app to return.
    Error(NonZero<u8>),
}

impl AppExit {
    /// Creates a [`AppExit::Error`] with an error code of 1.
    #[must_use]
    pub const fn error() -> Self {
        Self::Error(NonZero::<u8>::MIN)
    }

    /// Returns `true` if `self` is a [`AppExit::Success`].
    #[must_use]
    pub const fn is_success(&self) -> bool {
        matches!(self, AppExit::Success)
    }

    /// Returns `true` if `self` is a [`AppExit::Error`].
    #[must_use]
    pub const fn is_error(&self) -> bool {
        matches!(self, AppExit::Error(_))
    }

    /// Creates a [`AppExit`] from a code.
    ///
    /// When `code` is 0 a [`AppExit::Success`] is constructed otherwise a
    /// [`AppExit::Error`] is constructed.
    #[must_use]
    pub const fn from_code(code: u8) -> Self {
        match NonZero::<u8>::new(code) {
            Some(code) => Self::Error(code),
            None => Self::Success,
        }
    }
}

impl From<u8> for AppExit {
    fn from(value: u8) -> Self {
        Self::from_code(value)
    }
}

#[cfg(feature = "std")]
impl Termination for AppExit {
    fn report(self) -> ExitCode {
        match self {
            AppExit::Success => ExitCode::SUCCESS,
            // We leave logging an error to our users
            AppExit::Error(value) => ExitCode::from(value.get()),
        }
    }
}

#[cfg(test)]
mod tests {
    use core::marker::PhantomData;
    use std::sync::Mutex;

    use bevy_ecs::{
        change_detection::{DetectChanges, ResMut},
        component::Component,
        entity::Entity,
        lifecycle::RemovedComponents,
        message::{Message, MessageWriter, Messages},
        query::With,
        resource::Resource,
        schedule::{IntoScheduleConfigs, ScheduleLabel},
        system::{Commands, Query},
        world::{FromWorld, World},
    };

    use crate::{App, AppExit, Bevy, Plugin, Update};

    struct PluginA;
    impl Plugin for PluginA {
        fn build(&self, _app: &mut Bevy) {}
    }
    struct PluginB;
    impl Plugin for PluginB {
        fn build(&self, _app: &mut Bevy) {}
    }
    struct PluginC<T>(T);
    impl<T: Send + Sync + 'static> Plugin for PluginC<T> {
        fn build(&self, _app: &mut Bevy) {}
    }
    struct PluginD;
    impl Plugin for PluginD {
        fn build(&self, _app: &mut Bevy) {}
        fn is_unique(&self) -> bool {
            false
        }
    }

    struct PluginE;

    impl Plugin for PluginE {
        fn build(&self, _app: &mut Bevy) {}

        fn finish(&self, bevy: &mut Bevy) {
            if app.is_plugin_added::<PluginA>() {
                panic!("cannot run if PluginA is already registered");
            }
        }
    }

    struct PluginF;

    impl Plugin for PluginF {
        fn build(&self, _app: &mut Bevy) {}

        fn finish(&self, bevy: &mut Bevy) {
            // Ensure other plugins are available during finish
            assert_eq!(
                app.is_plugin_added::<PluginA>(),
                !app.get_added_plugins::<PluginA>().is_empty(),
            );
        }

        fn cleanup(&self, bevy: &mut Bevy) {
            // Ensure other plugins are available during finish
            assert_eq!(
                app.is_plugin_added::<PluginA>(),
                !app.get_added_plugins::<PluginA>().is_empty(),
            );
        }
    }

    struct PluginG;

    impl Plugin for PluginG {
        fn build(&self, _app: &mut Bevy) {}

        fn finish(&self, bevy: &mut Bevy) {
            app.add_plugins(PluginB);
        }
    }

    #[test]
    fn can_add_two_plugins() {
        Bevy::new().add_plugins((PluginA, PluginB));
    }

    #[test]
    #[should_panic]
    fn cant_add_twice_the_same_plugin() {
        Bevy::new().add_plugins((PluginA, PluginA));
    }

    #[test]
    fn can_add_twice_the_same_plugin_with_different_type_param() {
        Bevy::new().add_plugins((PluginC(0), PluginC(true)));
    }

    #[test]
    fn can_add_twice_the_same_plugin_not_unique() {
        Bevy::new().add_plugins((PluginD, PluginD));
    }

    #[test]
    #[should_panic]
    fn cant_call_app_run_from_plugin_build() {
        struct PluginRun;
        struct InnerPlugin;
        impl Plugin for InnerPlugin {
            fn build(&self, _: &mut Bevy) {}
        }
        impl Plugin for PluginRun {
            fn build(&self, bevy: &mut Bevy) {
                app.add_plugins(InnerPlugin).run();
            }
        }
        Bevy::new().add_plugins(PluginRun);
    }

    #[derive(ScheduleLabel, Hash, Clone, PartialEq, Eq, Debug)]
    struct EnterMainMenu;

    #[derive(Component)]
    struct A;

    fn bar(mut commands: Commands) {
        commands.spawn(A);
    }

    fn foo(mut commands: Commands) {
        commands.spawn(A);
    }

    #[test]
    fn add_systems_should_create_schedule_if_it_does_not_exist() {
        let mut app = Bevy::new();
        app.add_systems(EnterMainMenu, (foo, bar));

        app.world_mut().run_schedule(EnterMainMenu);
        assert_eq!(app.world_mut().query::<&A>().query(app.world()).count(), 2);
    }

    #[test]
    #[should_panic]
    fn test_is_plugin_added_works_during_finish() {
        let mut app = Bevy::new();
        app.add_plugins(PluginA);
        app.add_plugins(PluginE);
        app.finish();
    }

    #[test]
    fn test_get_added_plugins_works_during_finish_and_cleanup() {
        let mut app = Bevy::new();
        app.add_plugins(PluginA);
        app.add_plugins(PluginF);
        app.finish();
    }

    #[test]
    fn test_adding_plugin_works_during_finish() {
        let mut app = Bevy::new();
        app.add_plugins(PluginA);
        app.add_plugins(PluginG);
        app.finish();
        assert_eq!(
            app.main().plugin_registry[0].name(),
            "bevy_app::main_schedule::MainSchedulePlugin"
        );
        assert_eq!(
            app.main().plugin_registry[1].name(),
            "bevy_app::app::tests::PluginA"
        );
        assert_eq!(
            app.main().plugin_registry[2].name(),
            "bevy_app::app::tests::PluginG"
        );
        // PluginG adds PluginB during finish
        assert_eq!(
            app.main().plugin_registry[3].name(),
            "bevy_app::app::tests::PluginB"
        );
    }

    #[test]
    fn test_derive_app_label() {
        use super::AppLabel;

        #[derive(AppLabel, Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
        struct UnitLabel;

        #[derive(AppLabel, Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
        struct TupleLabel(u32, u32);

        #[derive(AppLabel, Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
        struct StructLabel {
            a: u32,
            b: u32,
        }

        #[expect(
            dead_code,
            reason = "This struct is used as a compilation test to test the derive macros, and as such is intentionally never constructed."
        )]
        #[derive(AppLabel, Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
        struct EmptyTupleLabel();

        #[expect(
            dead_code,
            reason = "This struct is used as a compilation test to test the derive macros, and as such is intentionally never constructed."
        )]
        #[derive(AppLabel, Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
        struct EmptyStructLabel {}

        #[derive(AppLabel, Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
        enum EnumLabel {
            #[default]
            Unit,
            Tuple(u32, u32),
            Struct {
                a: u32,
                b: u32,
            },
        }

        #[derive(AppLabel, Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
        struct GenericLabel<T>(PhantomData<T>);

        assert_eq!(UnitLabel.intern(), UnitLabel.intern());
        assert_eq!(EnumLabel::Unit.intern(), EnumLabel::Unit.intern());
        assert_ne!(UnitLabel.intern(), EnumLabel::Unit.intern());
        assert_ne!(UnitLabel.intern(), TupleLabel(0, 0).intern());
        assert_ne!(EnumLabel::Unit.intern(), EnumLabel::Tuple(0, 0).intern());

        assert_eq!(TupleLabel(0, 0).intern(), TupleLabel(0, 0).intern());
        assert_eq!(
            EnumLabel::Tuple(0, 0).intern(),
            EnumLabel::Tuple(0, 0).intern()
        );
        assert_ne!(TupleLabel(0, 0).intern(), TupleLabel(0, 1).intern());
        assert_ne!(
            EnumLabel::Tuple(0, 0).intern(),
            EnumLabel::Tuple(0, 1).intern()
        );
        assert_ne!(TupleLabel(0, 0).intern(), EnumLabel::Tuple(0, 0).intern());
        assert_ne!(
            TupleLabel(0, 0).intern(),
            StructLabel { a: 0, b: 0 }.intern()
        );
        assert_ne!(
            EnumLabel::Tuple(0, 0).intern(),
            EnumLabel::Struct { a: 0, b: 0 }.intern()
        );

        assert_eq!(
            StructLabel { a: 0, b: 0 }.intern(),
            StructLabel { a: 0, b: 0 }.intern()
        );
        assert_eq!(
            EnumLabel::Struct { a: 0, b: 0 }.intern(),
            EnumLabel::Struct { a: 0, b: 0 }.intern()
        );
        assert_ne!(
            StructLabel { a: 0, b: 0 }.intern(),
            StructLabel { a: 0, b: 1 }.intern()
        );
        assert_ne!(
            EnumLabel::Struct { a: 0, b: 0 }.intern(),
            EnumLabel::Struct { a: 0, b: 1 }.intern()
        );
        assert_ne!(
            StructLabel { a: 0, b: 0 }.intern(),
            EnumLabel::Struct { a: 0, b: 0 }.intern()
        );
        assert_ne!(
            StructLabel { a: 0, b: 0 }.intern(),
            EnumLabel::Struct { a: 0, b: 0 }.intern()
        );
        assert_ne!(StructLabel { a: 0, b: 0 }.intern(), UnitLabel.intern(),);
        assert_ne!(
            EnumLabel::Struct { a: 0, b: 0 }.intern(),
            EnumLabel::Unit.intern()
        );

        assert_eq!(
            GenericLabel::<u32>(PhantomData).intern(),
            GenericLabel::<u32>(PhantomData).intern()
        );
        assert_ne!(
            GenericLabel::<u32>(PhantomData).intern(),
            GenericLabel::<u64>(PhantomData).intern()
        );
    }

    #[test]
    fn test_update_clears_trackers_once() {
        #[derive(Component, Copy, Clone)]
        struct Foo;

        let mut app = Bevy::new();
        app.world_mut().spawn_batch(core::iter::repeat_n(Foo, 5));

        fn despawn_one_foo(mut commands: Commands, foos: Query<Entity, With<Foo>>) {
            if let Some(e) = foos.iter().next() {
                commands.entity(e).despawn();
            };
        }
        fn check_despawns(mut removed_foos: RemovedComponents<Foo>) {
            let mut despawn_count = 0;
            for _ in removed_foos.read() {
                despawn_count += 1;
            }

            assert_eq!(despawn_count, 2);
        }

        app.add_systems(Update, despawn_one_foo);
        app.update(); // Frame 0
        app.update(); // Frame 1
        app.add_systems(Update, check_despawns.after(despawn_one_foo));
        app.update(); // Should see despawns from frames 1 & 2, but not frame 0
    }

    #[test]
    fn test_extract_sees_changes() {
        use super::AppLabel;

        #[derive(AppLabel, Clone, Copy, Hash, PartialEq, Eq, Debug)]
        struct MyApp;

        #[derive(Resource)]
        struct Foo(usize);

        let mut app = Bevy::new();
        app.world_mut().insert_resource(Foo(0));
        app.add_systems(Update, |mut foo: ResMut<Foo>| {
            foo.0 += 1;
        });

        let mut app = App::new();
        app.set_extract(|main_world, _sub_world| {
            assert!(main_world.get_resource_ref::<Foo>().unwrap().is_changed());
        });

        app.insert_app(MyApp, app);

        app.update();
    }

    #[test]
    fn runner_returns_correct_exit_code() {
        fn raise_exits(mut exits: MessageWriter<AppExit>) {
            // Exit codes chosen by a fair dice roll.
            // Unlikely to overlap with default values.
            exits.write(AppExit::Success);
            exits.write(AppExit::from_code(4));
            exits.write(AppExit::from_code(73));
        }

        let exit = Bevy::new().add_systems(Update, raise_exits).run();

        assert_eq!(exit, AppExit::from_code(4));
    }

    /// Custom runners should be in charge of when `app::update` gets called as they may need to
    /// coordinate some state.
    /// bug: <https://github.com/bevyengine/bevy/issues/10385>
    /// fix: <https://github.com/bevyengine/bevy/pull/10389>
    #[test]
    fn regression_test_10385() {
        use super::{Res, Resource};
        use crate::PreUpdate;

        #[derive(Resource)]
        struct MyState {}

        fn my_runner(mut app: Bevy) -> AppExit {
            let my_state = MyState {};
            app.world_mut().insert_resource(my_state);

            for _ in 0..5 {
                app.update();
            }

            AppExit::Success
        }

        fn my_system(_: Res<MyState>) {
            // access state during app update
        }

        // Should not panic due to missing resource
        Bevy::new()
            .set_runner(my_runner)
            .add_systems(PreUpdate, my_system)
            .run();
    }

    #[test]
    fn app_exit_size() {
        // There wont be many of them so the size isn't an issue but
        // it's nice they're so small let's keep it that way.
        assert_eq!(size_of::<AppExit>(), size_of::<u8>());
    }

    #[test]
    fn initializing_resources_from_world() {
        #[derive(Resource)]
        struct TestResource;
        impl FromWorld for TestResource {
            fn from_world(_world: &mut World) -> Self {
                TestResource
            }
        }

        #[derive(Resource)]
        struct NonSendTestResource {
            _marker: PhantomData<Mutex<()>>,
        }
        impl FromWorld for NonSendTestResource {
            fn from_world(_world: &mut World) -> Self {
                NonSendTestResource {
                    _marker: PhantomData,
                }
            }
        }

        Bevy::new()
            .init_non_send::<NonSendTestResource>()
            .init_resource::<TestResource>();
    }

    #[test]
    /// Plugin should not be considered inserted while it's being built
    ///
    /// bug: <https://github.com/bevyengine/bevy/issues/13815>
    fn plugin_should_not_be_added_during_build_time() {
        pub struct Foo;

        impl Plugin for Foo {
            fn build(&self, bevy: &mut Bevy) {
                assert!(!app.is_plugin_added::<Self>());
            }
        }

        Bevy::new().add_plugins(Foo);
    }
    #[test]
    fn events_should_be_updated_once_per_update() {
        #[derive(Message, Clone)]
        struct TestMessage;

        let mut app = Bevy::new();
        app.add_message::<TestMessage>();

        // Starts empty
        let test_messages = app.world().resource::<Messages<TestMessage>>();
        assert_eq!(test_messages.len(), 0);
        assert_eq!(test_messages.iter_current_update_messages().count(), 0);
        app.update();

        // Sending one event
        app.world_mut().write_message(TestMessage);

        let test_events = app.world().resource::<Messages<TestMessage>>();
        assert_eq!(test_events.len(), 1);
        assert_eq!(test_events.iter_current_update_messages().count(), 1);
        app.update();

        // Sending two events on the next frame
        app.world_mut().write_message(TestMessage);
        app.world_mut().write_message(TestMessage);

        let test_events = app.world().resource::<Messages<TestMessage>>();
        assert_eq!(test_events.len(), 3); // Events are double-buffered, so we see 1 + 2 = 3
        assert_eq!(test_events.iter_current_update_messages().count(), 2);
        app.update();

        // Sending zero events
        let test_events = app.world().resource::<Messages<TestMessage>>();
        assert_eq!(test_events.len(), 2); // Events are double-buffered, so we see 2 + 0 = 2
        assert_eq!(test_events.iter_current_update_messages().count(), 0);
    }

    #[test]
    fn auto_despawn_unused_registered_systems() {
        let mut app = Bevy::new();

        fn my_system() {}

        let handle = app.register_tracked_system(my_system);
        let entity = handle.entity();

        app.update();
        assert!(app.world().get_entity(entity).is_ok());

        drop(handle);
        app.update();
        assert!(app.world().get_entity(entity).is_err());
    }
}
