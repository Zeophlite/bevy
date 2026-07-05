use crate::{AppLabel, Bevy, InternedAppLabel, Plugin, Plugins, PluginsState};
use alloc::{boxed::Box, string::String, vec::Vec};
use bevy_ecs::{
    message::MessageRegistry,
    observer::IntoObserver,
    prelude::*,
    schedule::{
        InternedScheduleLabel, InternedSystemSet, ScheduleBuildSettings, ScheduleCleanupPolicy,
        ScheduleError, ScheduleLabel,
    },
    system::{ScheduleSystem, SystemId, SystemInput},
};
use bevy_platform::collections::{HashMap, HashSet};
use core::fmt::Debug;

#[cfg(feature = "trace")]
use tracing::{info_span, warn};

type ExtractFn = Box<dyn FnMut(&mut World, &mut World) + Send>;

/// A secondary application with its own [`World`]. These can run independently of each other.
///
/// These are useful for situations where certain processes (e.g. a render thread) need to be kept
/// separate from the main application.
///
/// # Example
///
/// ```
/// # use bevy_app::{App, AppLabel, App, Main};
/// # use bevy_ecs::prelude::*;
/// # use bevy_ecs::schedule::ScheduleLabel;
///
/// #[derive(Resource, Default)]
/// struct Val(pub i32);
///
/// #[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, AppLabel)]
/// struct ExampleApp;
///
/// // Create an app with a certain resource.
/// let mut app = App::new();
/// app.insert_resource(Val(10));
///
/// // Create a sub-app with the same resource and a single schedule.
/// let mut app = App::new();
/// app.update_schedule = Some(Main.intern());
/// app.insert_resource(Val(100));
///
/// // Setup an extract function to copy the resource's value in the main world.
/// app.set_extract(|main_world, sub_world| {
///     sub_world.resource_mut::<Val>().0 = main_world.resource::<Val>().0;
/// });
///
/// // Schedule a system that will verify extraction is working.
/// app.add_systems(Main, |counter: Res<Val>| {
///     // The value will be copied during extraction, so we should see 10 instead of 100.
///     assert_eq!(counter.0, 10);
/// });
///
/// // Add the sub-app to the main app.
/// app.insert_app(ExampleApp, app);
///
/// // Update the application once (using the default runner).
/// app.run();
/// ```
pub struct App {
    /// The data of this application.
    world: World,
    /// List of plugins that have been added.
    pub(crate) plugin_registry: Vec<Box<dyn Plugin>>,
    /// The names of plugins that have been added to this app. (used to track duplicates and
    /// already-registered plugins)
    pub(crate) plugin_names: HashSet<String>,
    /// Panics if an update is attempted while plugins are building.
    pub(crate) plugin_build_depth: usize,
    pub(crate) plugins_state: PluginsState,
    /// The schedule that will be run by [`update`](Self::update).
    pub update_schedule: Option<InternedScheduleLabel>,
    /// A function that gives mutable access to two app worlds. This is primarily
    /// intended for copying data from the main world to secondary worlds.
    extract: Option<ExtractFn>,
}

impl Debug for App {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "App")
    }
}

impl Default for App {
    fn default() -> Self {
        let mut world = World::new();
        world.init_resource::<Schedules>();
        Self {
            world,
            plugin_registry: Vec::default(),
            plugin_names: HashSet::default(),
            plugin_build_depth: 0,
            plugins_state: PluginsState::Adding,
            update_schedule: None,
            extract: None,
        }
    }
}

impl App {
    /// Returns a default, empty [`App`].
    pub fn new() -> Self {
        Self::default()
    }

    /// This method is a workaround. Each [`App`] can have its own plugins, but [`Plugin`]
    /// works on an [`App`] as a whole.
    fn run_as_app<F>(&mut self, f: F)
    where
        F: FnOnce(&mut Bevy),
    {
        let mut app = Bevy::empty();
        core::mem::swap(self, &mut app.apps.main);
        f(&mut app);
        core::mem::swap(self, &mut app.apps.main);
    }

    /// Returns a reference to the [`World`].
    pub fn world(&self) -> &World {
        &self.world
    }

    /// Returns a mutable reference to the [`World`].
    pub fn world_mut(&mut self) -> &mut World {
        &mut self.world
    }

    /// A
    pub fn register_required_components<T: Component, R: Component + Default>(
        &mut self,
    ) -> &mut Self {
        self.world_mut().register_required_components::<T, R>();
        self
    }

    /// A
    pub fn register_required_components_with<T: Component, R: Component>(
        &mut self,
        constructor: fn() -> R,
    ) -> &mut Self {
        self.world_mut()
            .register_required_components_with::<T, R>(constructor);
        self
    }

    /// Runs the default schedule.
    ///
    /// Does not clear internal trackers used for change detection.
    pub fn run_default_schedule(&mut self) {
        if self.is_building_plugins() {
            panic!("App::update() was called while a plugin was building.");
        }

        if let Some(label) = self.update_schedule {
            self.world.run_schedule(label);
        }
    }

    /// Runs the default schedule and updates internal component trackers.
    pub fn update(&mut self) {
        self.run_default_schedule();
        self.world.clear_trackers();
    }

    /// Extracts data from `world` into the app's world using the registered extract method.
    ///
    /// **Note:** There is no default extract method. Calling `extract` does nothing if
    /// [`set_extract`](Self::set_extract) has not been called.
    pub fn extract(&mut self, world: &mut World) {
        if let Some(f) = self.extract.as_mut() {
            f(world, &mut self.world);
        }
    }

    /// Sets the method that will be called by [`extract`](Self::extract).
    ///
    /// The first argument is the `World` to extract data from, the second argument is the app `World`.
    pub fn set_extract<F>(&mut self, extract: F) -> &mut Self
    where
        F: FnMut(&mut World, &mut World) + Send + 'static,
    {
        self.extract = Some(Box::new(extract));
        self
    }

    /// Take the function that will be called by [`extract`](Self::extract) out of the app, if any was set,
    /// and replace it with `None`.
    ///
    /// If you use Bevy, `bevy_render` will set a default extract function used to extract data from
    /// the main world into the render world as part of the Extract phase. In that case, you cannot replace
    /// it with your own function. Instead, take the Bevy default function with this, and install your own
    /// instead which calls the Bevy default.
    ///
    /// ```
    /// # use bevy_app::App;
    /// # let mut app = App::new();
    /// let mut default_fn = app.take_extract();
    /// app.set_extract(move |main, render| {
    ///     // Do pre-extract custom logic
    ///     // [...]
    ///
    ///     // Call Bevy's default, which executes the Extract phase
    ///     if let Some(f) = default_fn.as_mut() {
    ///         f(main, render);
    ///     }
    ///
    ///     // Do post-extract custom logic
    ///     // [...]
    /// });
    /// ```
    pub fn take_extract(&mut self) -> Option<ExtractFn> {
        self.extract.take()
    }

    /// Inserts the [`Resource`] into the app, overwriting any existing resource of the same type.
    ///
    /// There is also an [`init_resource`](Self::init_resource) for resources that have
    /// [`Default`] or [`FromWorld`] implementations.
    ///
    /// # Examples
    ///
    /// ```
    /// # use bevy_app::prelude::*;
    /// # use bevy_ecs::prelude::*;
    /// #
    /// #[derive(Resource)]
    /// struct MyCounter {
    ///     counter: usize,
    /// }
    ///
    /// App::new()
    ///    .insert_resource(MyCounter { counter: 0 });
    /// ```
    pub fn insert_resource<R: Resource>(&mut self, resource: R) -> &mut Self {
        self.world.insert_resource(resource);
        self
    }

    /// Inserts the [`Resource`], initialized with its default value, into the app,
    /// if there is no existing instance of `R`.
    ///
    /// `R` must implement [`FromWorld`].
    /// If `R` implements [`Default`], [`FromWorld`] will be automatically implemented and
    /// initialize the [`Resource`] with [`Default::default`].
    ///
    /// # Examples
    ///
    /// ```
    /// # use bevy_app::prelude::*;
    /// # use bevy_ecs::prelude::*;
    /// #
    /// #[derive(Resource)]
    /// struct MyCounter {
    ///     counter: usize,
    /// }
    ///
    /// impl Default for MyCounter {
    ///     fn default() -> MyCounter {
    ///         MyCounter {
    ///             counter: 100
    ///         }
    ///     }
    /// }
    ///
    /// App::new()
    ///     .init_resource::<MyCounter>();
    /// ```
    pub fn init_resource<R: Resource + FromWorld>(&mut self) -> &mut Self {
        self.world.init_resource::<R>();
        self
    }

    /// Adds one or more systems to the given schedule in this app's [`Schedules`].
    ///
    /// # Examples
    ///
    /// ```
    /// # use bevy_app::prelude::*;
    /// # use bevy_ecs::prelude::*;
    /// #
    /// # let mut app = App::new();
    /// # fn system_a() {}
    /// # fn system_b() {}
    /// # fn system_c() {}
    /// # fn should_run() -> bool { true }
    /// #
    /// app.add_systems(Update, (system_a, system_b, system_c));
    /// app.add_systems(Update, (system_a, system_b).run_if(should_run));
    /// ```
    pub fn add_systems<M>(
        &mut self,
        schedule: impl ScheduleLabel,
        systems: impl IntoScheduleConfigs<ScheduleSystem, M>,
    ) -> &mut Self {
        let mut schedules = self.world.resource_mut::<Schedules>();
        schedules.add_systems(schedule, systems);

        self
    }

    /// Removes all systems in a [`SystemSet`]. This will cause the schedule to be rebuilt when
    /// the schedule is run again and can be slow. A [`ScheduleError`] is returned if the schedule needs to be
    /// [`Schedule::initialize`]'d or the `set` is not found.
    ///
    /// Note that this can remove all systems of a type if you pass
    /// the system to this function as systems implicitly create a set based
    /// on the system type.
    ///
    /// ## Example
    /// ```
    /// # use bevy_app::prelude::*;
    /// # use bevy_ecs::schedule::ScheduleCleanupPolicy;
    /// #
    /// # let mut app = App::new();
    /// # fn system_a() {}
    /// # fn system_b() {}
    /// #
    /// // add the system
    /// app.add_systems(Update, system_a);
    ///
    /// // remove the system
    /// app.remove_systems_in_set(Update, system_a, ScheduleCleanupPolicy::RemoveSystemsOnly);
    /// ```
    pub fn remove_systems_in_set<M>(
        &mut self,
        schedule: impl ScheduleLabel,
        set: impl IntoSystemSet<M>,
        policy: ScheduleCleanupPolicy,
    ) -> Result<usize, ScheduleError> {
        self.world.schedule_scope(schedule, |world, schedule| {
            schedule.remove_systems_in_set(set, world, policy)
        })
    }

    /// Registers a system and returns a [`SystemId`] so it can later be called by [`World::run_system`].
    ///
    /// It's possible to register the same systems more than once, they'll be stored separately.
    ///
    /// This is different from adding systems to a [`Schedule`] with [`App::add_systems`],
    /// because the [`SystemId`] that is returned can be used anywhere in the [`World`] to run the associated system.
    /// This allows for running systems in a push-based fashion.
    /// Using a [`Schedule`] is still preferred for most cases
    /// due to its better performance and ability to run non-conflicting systems simultaneously.
    pub fn register_system<I, O, M>(
        &mut self,
        system: impl IntoSystem<I, O, M> + 'static,
    ) -> SystemId<I, O>
    where
        I: SystemInput + 'static,
        O: 'static,
    {
        self.world.register_system(system)
    }

    /// Registers a system and returns a tracked [`SystemHandle`] so it can later
    /// be called by [`World::run_system`]. The system entity will be automatically
    /// queued for despawn when the last clone of the returned handle is dropped.
    ///
    /// See [`World::register_tracked_system`] for more details.
    ///
    /// [`SystemHandle`]: bevy_ecs::system::SystemHandle
    pub fn register_tracked_system<I, O, M>(
        &mut self,
        system: impl IntoSystem<I, O, M> + 'static,
    ) -> bevy_ecs::system::SystemHandle<I, O>
    where
        I: SystemInput + 'static,
        O: 'static,
    {
        self.world.register_tracked_system(system)
    }

    /// Configures a collection of system sets in the provided schedule, adding any sets that do not exist.
    #[track_caller]
    pub fn configure_sets<M>(
        &mut self,
        schedule: impl ScheduleLabel,
        sets: impl IntoScheduleConfigs<InternedSystemSet, M>,
    ) -> &mut Self {
        let mut schedules = self.world.resource_mut::<Schedules>();
        schedules.configure_sets(schedule, sets);
        self
    }

    /// Inserts a new `schedule` under the provided `label`, overwriting any existing
    /// schedule with the same label.
    pub fn add_schedule(&mut self, schedule: Schedule) -> &mut Self {
        let mut schedules = self.world.resource_mut::<Schedules>();
        let _old_schedule = schedules.insert(schedule);

        #[cfg(feature = "trace")]
        if let Some(schedule) = _old_schedule {
            warn!(
                "Schedule {:?} was re-inserted, all previous configuration has been removed",
                schedule.label()
            );
        }

        self
    }

    /// Initializes an empty `schedule` under the provided `label`, if it does not exist.
    ///
    /// See [`add_schedule`](Self::add_schedule) to insert an existing schedule.
    pub fn init_schedule(&mut self, label: impl ScheduleLabel) -> &mut Self {
        let label = label.intern();
        let mut schedules = self.world.resource_mut::<Schedules>();
        if !schedules.contains(label) {
            schedules.insert(Schedule::new(label));
        }
        self
    }

    /// Returns a reference to the [`Schedule`] with the provided `label` if it exists.
    pub fn get_schedule(&self, label: impl ScheduleLabel) -> Option<&Schedule> {
        let schedules = self.world.get_resource::<Schedules>()?;
        schedules.get(label)
    }

    /// Returns a mutable reference to the [`Schedule`] with the provided `label` if it exists.
    pub fn get_schedule_mut(&mut self, label: impl ScheduleLabel) -> Option<&mut Schedule> {
        let schedules = self.world.get_resource_mut::<Schedules>()?;
        // We must call `.into_inner` here because the borrow checker only understands reborrows
        // using ordinary references, not our `Mut` smart pointers.
        schedules.into_inner().get_mut(label)
    }

    /// Runs function `f` with the [`Schedule`] associated with `label`.
    ///
    /// **Note:** This will create the schedule if it does not already exist.
    pub fn edit_schedule(
        &mut self,
        label: impl ScheduleLabel,
        mut f: impl FnMut(&mut Schedule),
    ) -> &mut Self {
        let label = label.intern();
        let mut schedules = self.world.resource_mut::<Schedules>();
        if !schedules.contains(label) {
            schedules.insert(Schedule::new(label));
        }

        let schedule = schedules.get_mut(label).unwrap();
        f(schedule);

        self
    }

    /// Applies the provided [`ScheduleBuildSettings`] to all schedules.
    ///
    /// This mutates all currently present schedules, but does not apply to any custom schedules
    /// that might be added in the future.
    pub fn configure_schedules(
        &mut self,
        schedule_build_settings: ScheduleBuildSettings,
    ) -> &mut Self {
        self.world_mut()
            .resource_mut::<Schedules>()
            .configure_schedules(schedule_build_settings);
        self
    }

    /// When doing [ambiguity checking](ScheduleBuildSettings) this
    /// ignores systems that are ambiguous on [`Component`] T.
    ///
    /// This settings only applies to the main world. To apply this to other worlds call the
    /// [corresponding method](World::allow_ambiguous_component) on World
    ///
    /// ## Example
    ///
    /// ```
    /// # use bevy_app::prelude::*;
    /// # use bevy_ecs::prelude::*;
    /// # use bevy_ecs::schedule::{LogLevel, ScheduleBuildSettings};
    /// # use bevy_utils::default;
    ///
    /// #[derive(Component)]
    /// struct A;
    ///
    /// // these systems are ambiguous on A
    /// fn system_1(_: Query<&mut A>) {}
    /// fn system_2(_: Query<&A>) {}
    ///
    /// let mut app = App::new();
    /// app.configure_schedules(ScheduleBuildSettings {
    ///   ambiguity_detection: LogLevel::Error,
    ///   ..default()
    /// });
    ///
    /// app.add_systems(Update, ( system_1, system_2 ));
    /// app.allow_ambiguous_component::<A>();
    ///
    /// // running the app does not error.
    /// app.update();
    /// ```
    pub fn allow_ambiguous_component<T: Component>(&mut self) -> &mut Self {
        self.world_mut().allow_ambiguous_component::<T>();
        self
    }

    /// When doing [ambiguity checking](ScheduleBuildSettings) this
    /// ignores systems that are ambiguous on [`Resource`] T.
    ///
    /// This settings only applies to the main world. To apply this to other worlds call the
    /// [corresponding method](World::allow_ambiguous_resource) on World
    ///
    /// ## Example
    ///
    /// ```
    /// # use bevy_app::prelude::*;
    /// # use bevy_ecs::prelude::*;
    /// # use bevy_ecs::schedule::{LogLevel, ScheduleBuildSettings};
    /// # use bevy_utils::default;
    ///
    /// #[derive(Resource)]
    /// struct R;
    ///
    /// // these systems are ambiguous on R
    /// fn system_1(_: ResMut<R>) {}
    /// fn system_2(_: Res<R>) {}
    ///
    /// let mut app = App::new();
    /// app.configure_schedules(ScheduleBuildSettings {
    ///   ambiguity_detection: LogLevel::Error,
    ///   ..default()
    /// });
    /// app.insert_resource(R);
    ///
    /// app.add_systems(Update, ( system_1, system_2 ));
    /// app.allow_ambiguous_resource::<R>();
    ///
    /// // running the app does not error.
    /// app.update();
    /// ```
    pub fn allow_ambiguous_resource<T: Resource>(&mut self) -> &mut Self {
        self.world_mut().allow_ambiguous_resource::<T>();
        self
    }

    /// Suppress warnings and errors that would result from systems in these sets having ambiguities
    /// (conflicting access but indeterminate order) with systems in `set`.
    ///
    /// When possible, do this directly in the `.add_systems(Update, a.ambiguous_with(b))` call.
    /// However, sometimes two independent plugins `A` and `B` are reported as ambiguous, which you
    /// can only suppress as the consumer of both.
    #[track_caller]
    pub fn ignore_ambiguity<M1, M2, S1, S2>(
        &mut self,
        schedule: impl ScheduleLabel,
        a: S1,
        b: S2,
    ) -> &mut Self
    where
        S1: IntoSystemSet<M1>,
        S2: IntoSystemSet<M2>,
    {
        let schedule = schedule.intern();
        let mut schedules = self.world.resource_mut::<Schedules>();

        schedules.ignore_ambiguity(schedule, a, b);

        self
    }

    /// Spawns an [`Observer`] entity, which will watch for and respond to the given event.
    ///
    /// `observer` can be any system whose first parameter is [`On`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bevy_app::prelude::*;
    /// # use bevy_ecs::prelude::*;
    /// # use bevy_utils::default;
    /// #
    /// # let mut app = App::new();
    /// #
    /// # #[derive(Event)]
    /// # struct Party {
    /// #   friends_allowed: bool,
    /// # };
    /// #
    /// # #[derive(EntityEvent)]
    /// # struct Invite {
    /// #    entity: Entity,
    /// # }
    /// #
    /// # #[derive(Component)]
    /// # struct Friend;
    /// #
    ///
    /// app.add_observer(|event: On<Party>, friends: Query<Entity, With<Friend>>, mut commands: Commands| {
    ///     if event.friends_allowed {
    ///         for entity in friends.iter() {
    ///             commands.trigger(Invite { entity } );
    ///         }
    ///     }
    /// });
    /// ```
    pub fn add_observer<M>(&mut self, observer: impl IntoObserver<M>) -> &mut Self {
        self.world_mut().add_observer(observer);
        self
    }

    /// Initializes [`Message`] handling for `T` by inserting a message queue resource ([`Messages::<T>`])
    /// and scheduling an [`message_update_system`] in [`First`].
    ///
    /// See [`Messages`] for information on how to define messages.
    ///
    /// # Examples
    ///
    /// ```
    /// # use bevy_app::prelude::*;
    /// # use bevy_ecs::prelude::*;
    /// #
    /// # #[derive(Message)]
    /// # struct MyMessage;
    /// # let mut app = App::new();
    /// #
    /// app.add_message::<MyMessage>();
    /// ```
    pub fn add_message<T>(&mut self) -> &mut Self
    where
        T: Message,
    {
        if !self.world.contains_resource::<Messages<T>>() {
            MessageRegistry::register_message::<T>(self.world_mut());
        }

        self
    }

    /// See [`Bevy::add_plugins`].
    pub fn add_plugins<M>(&mut self, plugins: impl Plugins<M>) -> &mut Self {
        self.run_as_app(|app| plugins.add_to_bevy(app));
        self
    }

    /// Returns `true` if the [`Plugin`] has already been added.
    pub fn is_plugin_added<T>(&self) -> bool
    where
        T: Plugin,
    {
        self.plugin_names.contains(core::any::type_name::<T>())
    }

    /// Returns a vector of references to all plugins of type `T` that have been added.
    ///
    /// This can be used to read the settings of any existing plugins.
    /// This vector will be empty if no plugins of that type have been added.
    /// If multiple copies of the same plugin are added to the [`App`], they will be listed in insertion order in this vector.
    ///
    /// ```
    /// # use bevy_app::prelude::*;
    /// # #[derive(Default)]
    /// # struct ImagePlugin {
    /// #    default_sampler: bool,
    /// # }
    /// # impl Plugin for ImagePlugin {
    /// #    fn build(&self, app: &mut App) {}
    /// # }
    /// # let mut app = App::new();
    /// # app.add_plugins(ImagePlugin::default());
    /// let default_sampler = app.get_added_plugins::<ImagePlugin>()[0].default_sampler;
    /// ```
    pub fn get_added_plugins<T>(&self) -> Vec<&T>
    where
        T: Plugin,
    {
        self.plugin_registry
            .iter()
            .filter_map(|p| p.downcast_ref())
            .collect()
    }

    /// Returns `true` if there is no plugin in the middle of being built.
    pub(crate) fn is_building_plugins(&self) -> bool {
        self.plugin_build_depth > 0
    }

    /// Return the state of plugins.
    #[inline]
    pub fn plugins_state(&mut self) -> PluginsState {
        match self.plugins_state {
            PluginsState::Adding => {
                let mut state = PluginsState::Ready;
                let plugins = core::mem::take(&mut self.plugin_registry);
                self.run_as_app(|app| {
                    for plugin in &plugins {
                        if !plugin.ready(app) {
                            state = PluginsState::Adding;
                            return;
                        }
                    }
                });
                self.plugin_registry = plugins;
                state
            }
            state => state,
        }
    }

    /// Runs [`Plugin::finish`] for each plugin.
    pub fn finish(&mut self) {
        // do hokey pokey with a boxed zst plugin (doesn't allocate)
        let mut hokeypokey: Box<dyn Plugin> = Box::new(crate::HokeyPokey);
        for i in 0..self.plugin_registry.len() {
            core::mem::swap(&mut self.plugin_registry[i], &mut hokeypokey);
            #[cfg(feature = "trace")]
            let _plugin_finish_span =
                info_span!("plugin finish", plugin = hokeypokey.name()).entered();
            self.run_as_app(|app| {
                hokeypokey.finish(app);
            });
            core::mem::swap(&mut self.plugin_registry[i], &mut hokeypokey);
        }
        self.plugins_state = PluginsState::Finished;
    }

    /// Runs [`Plugin::cleanup`] for each plugin.
    pub fn cleanup(&mut self) {
        // do hokey pokey with a boxed zst plugin (doesn't allocate)
        let mut hokeypokey: Box<dyn Plugin> = Box::new(crate::HokeyPokey);
        for i in 0..self.plugin_registry.len() {
            core::mem::swap(&mut self.plugin_registry[i], &mut hokeypokey);
            #[cfg(feature = "trace")]
            let _plugin_cleanup_span =
                info_span!("plugin cleanup", plugin = hokeypokey.name()).entered();
            self.run_as_app(|app| {
                hokeypokey.cleanup(app);
            });
            core::mem::swap(&mut self.plugin_registry[i], &mut hokeypokey);
        }
        self.plugins_state = PluginsState::Cleaned;
    }

    /// Registers the type `T` in the [`AppTypeRegistry`] resource,
    /// adding reflect data as specified in the [`Reflect`] derive:
    /// ```ignore (No serde "derive" feature)
    /// #[derive(Component, Serialize, Deserialize, Reflect)]
    /// #[reflect(Component, Serialize, Deserialize)] // will register ReflectComponent, ReflectSerialize, ReflectDeserialize
    /// ```
    ///
    /// See [`bevy_reflect::TypeRegistry::register`] for more information.
    #[cfg(feature = "bevy_reflect")]
    pub fn register_type<T: bevy_reflect::GetTypeRegistration>(&mut self) -> &mut Self {
        let registry = self.world.resource_mut::<AppTypeRegistry>();
        registry.write().register::<T>();
        self
    }

    /// Associates type data `D` with type `T` in the [`AppTypeRegistry`] resource.
    ///
    /// Most of the time [`register_type`](Self::register_type) can be used instead to register a
    /// type you derived [`Reflect`] for. However, in cases where you want to
    /// add a piece of type data that was not included in the list of `#[reflect(...)]` type data in
    /// the derive, or where the type is generic and cannot register e.g. `ReflectSerialize`
    /// unconditionally without knowing the specific type parameters, this method can be used to
    /// insert additional type data.
    ///
    /// # Example
    /// ```
    /// use bevy_app::App;
    /// use bevy_reflect::{ReflectSerialize, ReflectDeserialize};
    ///
    /// App::new()
    ///     .register_type::<Option<String>>()
    ///     .register_type_data::<Option<String>, ReflectSerialize>()
    ///     .register_type_data::<Option<String>, ReflectDeserialize>();
    /// ```
    ///
    /// See [`bevy_reflect::TypeRegistry::register_type_data`].
    #[cfg(feature = "bevy_reflect")]
    pub fn register_type_data<
        T: bevy_reflect::Reflect + bevy_reflect::TypePath,
        D: bevy_reflect::CreateTypeData<T>,
    >(
        &mut self,
    ) -> &mut Self {
        let registry = self.world.resource_mut::<AppTypeRegistry>();
        registry.write().register_type_data::<T, D>();
        self
    }

    /// Registers a fallible conversion from type T to U with the reflection
    /// system.
    ///
    /// The supplied closure is expected to produce a value of type U, given an
    /// instance of type T. If the conversion fails, the closure should return
    /// the input value, wrapped in an `Err` variant.
    ///
    /// # Example
    /// ```
    /// use bevy_app::App;
    ///
    /// App::new()
    ///     .register_type::<i32>()
    ///     .register_type::<String>()
    ///     .register_type_conversion::<i32, String, _>(|n| Ok(n.to_string()));
    /// ```
    ///
    /// See [`bevy_reflect::TypeRegistry::register_type_conversion`].
    #[cfg(feature = "bevy_reflect")]
    pub fn register_type_conversion<T, U, F>(&mut self, function: F) -> &mut Self
    where
        T: bevy_reflect::Reflect + bevy_reflect::TypePath,
        U: bevy_reflect::Reflect + bevy_reflect::TypePath,
        F: Fn(T) -> Result<U, T> + Clone + Send + Sync + 'static,
    {
        let registry = self.world.resource_mut::<AppTypeRegistry>();
        registry
            .write()
            .register_type_conversion::<T, U, _>(function);
        self
    }

    /// Given types T and U, where `U: From<T>`, registers that conversion with
    /// the reflection system.
    ///
    /// # Example
    /// ```
    /// use bevy_app::App;
    ///
    /// App::new()
    ///     .register_type::<u8>()
    ///     .register_type::<u32>()
    ///     .register_into_type_conversion::<u8, u32>();
    /// ```
    ///
    /// See [`bevy_reflect::TypeRegistry::register_into_type_conversion`].
    #[cfg(feature = "bevy_reflect")]
    pub fn register_into_type_conversion<T, U>(&mut self) -> &mut Self
    where
        T: bevy_reflect::Reflect + bevy_reflect::TypePath,
        U: bevy_reflect::Reflect + bevy_reflect::TypePath + From<T>,
    {
        let registry = self.world.resource_mut::<AppTypeRegistry>();
        registry.write().register_into_type_conversion::<T, U>();
        self
    }

    /// Registers the given function into the [`AppFunctionRegistry`] resource.
    ///
    /// The given function will internally be stored as a [`DynamicFunction`]
    /// and mapped according to its [name].
    ///
    /// Because the function must have a name,
    /// anonymous functions (e.g. `|a: i32, b: i32| { a + b }`) and closures must instead
    /// be registered using [`register_function_with_name`] or converted to a [`DynamicFunction`]
    /// and named using [`DynamicFunction::with_name`].
    /// Failure to do so will result in a panic.
    ///
    /// Only types that implement [`IntoFunction`] may be registered via this method.
    ///
    /// See [`FunctionRegistry::register`] for more information.
    ///
    /// # Panics
    ///
    /// Panics if a function has already been registered with the given name
    /// or if the function is missing a name (such as when it is an anonymous function).
    ///
    /// # Examples
    ///
    /// ```
    /// use bevy_app::App;
    ///
    /// fn add(a: i32, b: i32) -> i32 {
    ///     a + b
    /// }
    ///
    /// App::new().register_function(add);
    /// ```
    ///
    /// Functions cannot be registered more than once.
    ///
    /// ```should_panic
    /// use bevy_app::App;
    ///
    /// fn add(a: i32, b: i32) -> i32 {
    ///     a + b
    /// }
    ///
    /// App::new()
    ///     .register_function(add)
    ///     // Panic! A function has already been registered with the name "my_function"
    ///     .register_function(add);
    /// ```
    ///
    /// Anonymous functions and closures should be registered using [`register_function_with_name`] or given a name using [`DynamicFunction::with_name`].
    ///
    /// ```should_panic
    /// use bevy_app::App;
    ///
    /// // Panic! Anonymous functions cannot be registered using `register_function`
    /// App::new().register_function(|a: i32, b: i32| a + b);
    /// ```
    ///
    /// [`register_function_with_name`]: Self::register_function_with_name
    /// [`DynamicFunction`]: bevy_reflect::func::DynamicFunction
    /// [name]: bevy_reflect::func::FunctionInfo::name
    /// [`DynamicFunction::with_name`]: bevy_reflect::func::DynamicFunction::with_name
    /// [`IntoFunction`]: bevy_reflect::func::IntoFunction
    /// [`FunctionRegistry::register`]: bevy_reflect::func::FunctionRegistry::register
    #[cfg(feature = "reflect_functions")]
    pub fn register_function<F, Marker>(&mut self, function: F) -> &mut Self
    where
        F: bevy_reflect::func::IntoFunction<'static, Marker> + 'static,
    {
        let registry = self.world.resource_mut::<AppFunctionRegistry>();
        registry.write().register(function).unwrap();
        self
    }

    /// Registers the given function or closure into the [`AppFunctionRegistry`] resource using the given name.
    ///
    /// To avoid conflicts, it's recommended to use a unique name for the function.
    /// This can be achieved by "namespacing" the function with a unique identifier,
    /// such as the name of your crate.
    ///
    /// For example, to register a function, `add`, from a crate, `my_crate`,
    /// you could use the name, `"my_crate::add"`.
    ///
    /// Another approach could be to use the [type name] of the function,
    /// however, it should be noted that anonymous functions do _not_ have unique type names.
    ///
    /// For named functions (e.g. `fn add(a: i32, b: i32) -> i32 { a + b }`) where a custom name is not needed,
    /// it's recommended to use [`register_function`] instead as the generated name is guaranteed to be unique.
    ///
    /// Only types that implement [`IntoFunction`] may be registered via this method.
    ///
    /// See [`FunctionRegistry::register_with_name`] for more information.
    ///
    /// # Panics
    ///
    /// Panics if a function has already been registered with the given name.
    ///
    /// # Examples
    ///
    /// ```
    /// use bevy_app::App;
    ///
    /// fn mul(a: i32, b: i32) -> i32 {
    ///     a * b
    /// }
    ///
    /// let div = |a: i32, b: i32| a / b;
    ///
    /// App::new()
    ///     // Registering an anonymous function with a unique name
    ///     .register_function_with_name("my_crate::add", |a: i32, b: i32| {
    ///         a + b
    ///     })
    ///     // Registering an existing function with its type name
    ///     .register_function_with_name(std::any::type_name_of_val(&mul), mul)
    ///     // Registering an existing function with a custom name
    ///     .register_function_with_name("my_crate::mul", mul)
    ///     // Be careful not to register anonymous functions with their type name.
    ///     // This code works but registers the function with a non-unique name like `foo::bar::{{closure}}`
    ///     .register_function_with_name(std::any::type_name_of_val(&div), div);
    /// ```
    ///
    /// Names must be unique.
    ///
    /// ```should_panic
    /// use bevy_app::App;
    ///
    /// fn one() {}
    /// fn two() {}
    ///
    /// App::new()
    ///     .register_function_with_name("my_function", one)
    ///     // Panic! A function has already been registered with the name "my_function"
    ///     .register_function_with_name("my_function", two);
    /// ```
    ///
    /// [type name]: std::any::type_name
    /// [`register_function`]: Self::register_function
    /// [`IntoFunction`]: bevy_reflect::func::IntoFunction
    /// [`FunctionRegistry::register_with_name`]: bevy_reflect::func::FunctionRegistry::register_with_name
    #[cfg(feature = "reflect_functions")]
    pub fn register_function_with_name<F, Marker>(
        &mut self,
        name: impl Into<alloc::borrow::Cow<'static, str>>,
        function: F,
    ) -> &mut Self
    where
        F: bevy_reflect::func::IntoFunction<'static, Marker> + 'static,
    {
        let registry = self.world.resource_mut::<AppFunctionRegistry>();
        registry.write().register_with_name(name, function).unwrap();
        self
    }
}

/// The collection of sub-apps that belong to an [`App`].
#[derive(Default)]
pub struct Apps {
    /// The primary sub-app that contains the "main" world.
    pub main: App,
    /// Other, labeled sub-apps.
    pub apps: HashMap<InternedAppLabel, App>,
}

impl Apps {
    /// Calls [`update`](App::update) for the main sub-app, and then calls
    /// [`extract`](App::extract) and [`update`](App::update) for the rest.
    pub fn update(&mut self) {
        #[cfg(feature = "trace")]
        let _bevy_update_span = info_span!("update").entered();
        {
            #[cfg(feature = "trace")]
            let _bevy_frame_update_span = info_span!("main app").entered();
            self.main.run_default_schedule();
        }
        for (_label, app) in self.apps.iter_mut() {
            #[cfg(feature = "trace")]
            let _app_span = info_span!("sub app", name = ?_label).entered();
            app.extract(&mut self.main.world);
            app.update();
        }

        self.main.world.clear_trackers();
    }

    /// Returns an iterator over the sub-apps (starting with the main one).
    pub fn iter(&self) -> impl Iterator<Item = &App> + '_ {
        core::iter::once(&self.main).chain(self.apps.values())
    }

    /// Returns a mutable iterator over the sub-apps (starting with the main one).
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut App> + '_ {
        core::iter::once(&mut self.main).chain(self.apps.values_mut())
    }

    /// Extract data from the main world into the [`App`] with the given label and perform an update if it exists.
    pub fn update_app_by_label(&mut self, label: impl AppLabel) {
        if let Some(app) = self.apps.get_mut(&label.intern()) {
            app.extract(&mut self.main.world);
            app.update();
        }
    }
}
