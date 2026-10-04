//! A source inspecting a separate running app over the Bevy Remote Protocol.
//!
//! Each remote entity is mirrored by a local proxy entity, spawned [`Disabled`] so that the
//! systems of the app running the inspector skip it. A proxy only carries what the entity tree
//! needs: the remote id, a resolved label and its parent proxy. Remote component values are never
//! inserted into the local world, so no local hook or observer runs for them.

use alloc::{
    string::{String, ToString},
    vec::Vec,
};
use core::time::Duration;

use bevy_dev_tools::inspection::label_resolution::{
    resolve_label, ComponentLabelData, LabelResolutionRegistry,
};
use bevy_ecs::{
    change_detection::Mut,
    component::{Component, ComponentId},
    entity::Entity,
    hierarchy::{ChildOf, Children},
    query::With,
    reflect::{AppTypeRegistry, ReflectComponent},
    resource::Resource,
    system::{Res, ResMut},
    world::World,
};
use bevy_log::{info, warn};
use bevy_platform::collections::HashMap;
use bevy_reflect::{prelude::ReflectDefault, Reflect, TypeRegistry};
use bevy_remote::{
    builtin_methods::{
        BrpAppInfoResponse, BrpQuery, BrpQueryFilter, BrpQueryParams, BrpQueryResponse,
        ComponentSelector, BRP_APP_INFO_METHOD, BRP_QUERY_METHOD,
    },
    client::{BrpClient, BrpClientError},
};
use bevy_tasks::{block_on, poll_once, Task};
use bevy_time::{Real, Time};
use serde_json::{Map, Value};

use crate::{
    details_panel::DetailsPanelSync, entity_tree::EntityTreeSync, InspectorSelection,
    InspectorSource,
};

const NAME: &str = "bevy_ecs::name::Name";
const CHILD_OF: &str = "bevy_ecs::hierarchy::ChildOf";
const IS_RESOURCE: &str = "bevy_ecs::resource::IsResource";

/// The shortest time between two `world.query` polls.
const POLL_INTERVAL: Duration = Duration::from_millis(500);
/// The time to wait before reconnecting after a failed request.
const RETRY_INTERVAL: Duration = Duration::from_secs(2);
/// The time after which a request without an answer is dropped and the connection marked failed.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(10);

/// The address of the remote app the inspector reads from.
#[derive(Debug, Clone, PartialEq, Eq, Reflect)]
#[reflect(Debug, Default, Clone, PartialEq)]
pub struct RemoteSource {
    /// The host the remote app serves the Bevy Remote Protocol on.
    pub host: String,
    /// The port the remote app serves the Bevy Remote Protocol on.
    pub port: u16,
}

impl Default for RemoteSource {
    fn default() -> Self {
        Self::localhost(15702)
    }
}

impl RemoteSource {
    /// A source reading from `host:port`.
    pub fn new(host: impl Into<String>, port: u16) -> Self {
        Self {
            host: host.into(),
            port,
        }
    }

    /// A source reading from `127.0.0.1:port`.
    pub fn localhost(port: u16) -> Self {
        Self::new("127.0.0.1", port)
    }
}

/// The state of the connection to the remote app.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub enum RemoteConnectionState {
    /// The inspector reads from the local world, or has not contacted the remote app yet.
    #[default]
    Disconnected,
    /// An `app.info` request is in flight.
    Connecting,
    /// The remote app answered `app.info` and is polled for its entities.
    Connected {
        /// The name the remote app reports.
        app_name: String,
        /// The Bevy version the remote app reports.
        bevy_version: String,
    },
    /// The last request failed. The connection is retried after a short delay.
    Failed(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RequestKind {
    Info,
    Query,
}

struct PendingCall {
    kind: RequestKind,
    task: Task<Result<Value, BrpClientError>>,
    started: Duration,
}

/// The connection to the remote app.
///
/// At most one poll is in flight at a time, so polls never pile up when the remote app answers
/// slower than the poll interval. Dropping the in-flight task cancels it.
#[derive(Resource, Default)]
pub struct RemoteConnection {
    /// The state of the connection.
    pub state: RemoteConnectionState,
    source: Option<RemoteSource>,
    client: Option<BrpClient>,
    pending: Option<PendingCall>,
    next_poll: Duration,
}

impl RemoteConnection {
    /// The client talking to the current source, if the inspector reads from a remote app.
    pub fn client(&self) -> Option<&BrpClient> {
        self.client.as_ref()
    }

    /// The source the connection was set up for.
    pub fn source(&self) -> Option<&RemoteSource> {
        self.source.as_ref()
    }

    /// Whether a poll is in flight.
    pub fn is_polling(&self) -> bool {
        self.pending.is_some()
    }

    fn fail(&mut self, error: String, now: Duration) {
        if !matches!(&self.state, RemoteConnectionState::Failed(previous) if *previous == error) {
            warn!("the remote inspector lost its connection: {error}");
        }
        self.state = RemoteConnectionState::Failed(error);
        self.next_poll = now + RETRY_INTERVAL;
    }
}

/// Contains the proxied remote worlds
#[derive(Resource, Debug, Default)]
pub struct RemoteWorlds {
    pub(crate) remote_main: World,
    // TODO: remote_render: World,
}


/// The entities of the remote world, as last reported by `world.query`.
#[derive(Resource, Debug, Default)]
pub struct RemoteSnapshot {
    rows: HashMap<Entity, Map<String, Value>>,
    order: Vec<Entity>,
    dirty: bool,
    revision: u64,
}

impl RemoteSnapshot {
    /// The serialized components of `remote`, keyed by full type path.
    pub fn components(&self, remote: Entity) -> Option<&Map<String, Value>> {
        self.rows.get(&remote)
    }

    /// The number of remote entities in the snapshot.
    pub fn len(&self) -> usize {
        self.order.len()
    }

    /// Whether the snapshot holds no entities.
    pub fn is_empty(&self) -> bool {
        self.order.is_empty()
    }

    fn set(&mut self, rows: BrpQueryResponse) {
        self.rows.clear();
        self.order.clear();
        for row in rows {
            if row.components.contains_key(IS_RESOURCE) || row.components.is_empty() {
                continue;
            }
            self.order.push(row.entity);
            self.rows
                .insert(row.entity, row.components.into_iter().collect());
        }
        self.dirty = true;
    }
}

/// A local proxy entity mirroring one entity of the remote world.
#[derive(Component, Debug, Clone, Copy, Reflect)]
#[reflect(Component, Debug, Clone)]
pub struct RemoteEntityProxy;

/// The label the entity tree shows for a proxy.
#[derive(Component, Debug, Default, Clone, PartialEq, Eq, Reflect)]
#[reflect(Component, Debug, Default, Clone, PartialEq)]
pub struct RemoteLabel(pub String);

/// The label the entity tree shows for `entity`, if it is a proxy.
pub(crate) fn proxy_label(world: &World, entity: Entity) -> Option<String> {
    // world.get_resource::<RemoteWorlds>().map(|rw|
        // rw.remote_main
        world
            .get::<RemoteLabel>(entity)
            .map(|label| label.0.clone())
            .or_else(|| Some(entity.to_string()))
    // )
}

/// Sets the connection up for the current [`InspectorSource`], clearing every proxy when it
/// changes.
pub fn sync_remote_source(
    mut selection: ResMut<InspectorSelection>,
    mut rw: ResMut<RemoteWorlds>,
    inspector_source: Res<InspectorSource>,
    mut connection: ResMut<RemoteConnection>,
    mut remote_snapshot: ResMut<RemoteSnapshot>,
    mut entity_tree_sync: ResMut<EntityTreeSync>,
    mut details_panel_sync: ResMut<DetailsPanelSync>,
) {
    let source = match &*inspector_source {
        InspectorSource::Remote(source) => Some(source.clone()),
        _ => None,
    };
    // if connection.source != source {
    //     return;
    // }
    // println!("sync_remote_source");

    clear_proxies(selection, &mut rw.remote_main);
    remote_snapshot.set(Vec::new());

    connection.pending = None;
    connection.state = RemoteConnectionState::Disconnected;
    connection.next_poll = Duration::ZERO;
    connection.client = source
        .as_ref()
        .map(|source| BrpClient::new(source.host.clone(), source.port));
    connection.source = source;

    entity_tree_sync.set_dirty();
    details_panel_sync.set_dirty();
}

/// Despawns every proxy, clearing the selection if it was one.
fn clear_proxies(mut selection: ResMut<InspectorSelection>, remote_world: &mut World) {
    let proxies: Vec<Entity> = remote_world
        .query_filtered::<Entity, With<RemoteEntityProxy>>()
        .iter(&remote_world)
        .collect();

    if selection
        .0
        .is_some_and(|selected| proxies.contains(&selected))
    {
        selection.0 = None;
    }
    for proxy in proxies {
        if let Ok(proxy) = remote_world.get_entity_mut(proxy) {
            proxy.despawn();
        }
    }
}

/// Polls the request in flight and starts the next one when it is due.
pub fn poll_remote_connection(
    time: Option<Res<Time<Real>>>,
    mut connection: ResMut<RemoteConnection>,
    mut snapshot: ResMut<RemoteSnapshot>,
) {
    let Some(client) = connection.client.clone() else {
        return;
    };
    let now = time.map(|time| time.elapsed()).unwrap_or_default();
    let connection = &mut *connection;

    if let Some(mut pending) = connection.pending.take() {
        match block_on(poll_once(&mut pending.task)) {
            Some(result) => {
                let elapsed = now.saturating_sub(pending.started);
                finish_call(
                    connection,
                    &mut snapshot,
                    pending.kind,
                    result,
                    now,
                    elapsed,
                );
            }
            None if now.saturating_sub(pending.started) >= REQUEST_TIMEOUT => {
                connection.fail("the remote app did not answer in time".to_string(), now);
            }
            None => {
                connection.pending = Some(pending);
                return;
            }
        }
    }

    if connection.pending.is_some() || now < connection.next_poll {
        return;
    }

    let (kind, task) = match connection.state {
        RemoteConnectionState::Connected { .. } => (
            RequestKind::Query,
            client.spawn_call(BRP_QUERY_METHOD, Some(query_params())),
        ),
        _ => {
            if connection.state == RemoteConnectionState::Disconnected {
                connection.state = RemoteConnectionState::Connecting;
            }
            (
                RequestKind::Info,
                client.spawn_call(BRP_APP_INFO_METHOD, None),
            )
        }
    };
    connection.pending = Some(PendingCall {
        kind,
        task,
        started: now,
    });
}

fn query_params() -> Value {
    let params = BrpQueryParams {
        data: BrpQuery {
            components: Vec::new(),
            option: ComponentSelector::All,
            has: Vec::new(),
        },
        filter: BrpQueryFilter {
            without: alloc::vec![IS_RESOURCE.to_string()],
            with: Vec::new(),
        },
        strict: false,
    };
    serde_json::to_value(params).unwrap_or_default()
}

fn finish_call(
    connection: &mut RemoteConnection,
    snapshot: &mut RemoteSnapshot,
    kind: RequestKind,
    result: Result<Value, BrpClientError>,
    now: Duration,
    elapsed: Duration,
) {
    // println!("finish_call");
    let value = match result {
        Ok(value) => value,
        Err(error) => {
            connection.fail(error.to_string(), now);
            return;
        }
    };

    match kind {
        RequestKind::Info => match serde_json::from_value::<BrpAppInfoResponse>(value) {
            Ok(info) => {
                info!(
                    "the remote inspector connected to {} ({})",
                    info.app_name, info.bevy_version
                );
                connection.state = RemoteConnectionState::Connected {
                    app_name: info.app_name,
                    bevy_version: info.bevy_version,
                };
                connection.next_poll = now;
            }
            Err(error) => connection.fail(error.to_string(), now),
        },
        RequestKind::Query => match serde_json::from_value::<BrpQueryResponse>(value) {
            Ok(rows) => {
                snapshot.set(rows);
                connection.next_poll = now + POLL_INTERVAL.max(elapsed * 2);
            }
            Err(error) => connection.fail(error.to_string(), now),
        },
    }
}

/// Spawns, reparents, relabels and despawns the proxies so that they match the latest snapshot.
pub fn apply_remote_snapshot(world: &mut World) {
    if !world.resource::<RemoteSnapshot>().dirty {
        return;
    }
    // println!("apply_remote_snapshot");

    world.resource_scope(|world, mut snapshot: Mut<RemoteSnapshot>| {
        snapshot.dirty = false;
        snapshot.revision += 1;

        world.resource_scope(|world, mut rw: Mut<RemoteWorlds>| {
            // let mut rw = world.resource_mut::<RemoteWorlds>();
            let remote_world = &mut rw.remote_main;

            despawn_vanished(world, remote_world, &snapshot);
            spawn_missing(world, remote_world, &snapshot);
            apply_hierarchy(world, remote_world, &snapshot);
            apply_labels(world, remote_world, &snapshot);
        })
    });
    world.resource_mut::<EntityTreeSync>().set_dirty();
    world.resource_mut::<DetailsPanelSync>().set_dirty();
}

fn despawn_vanished(_world: &mut World, remote_world: &mut World, snapshot: &RemoteSnapshot) {
    let stale: Vec<Entity> = remote_world
        .query_filtered::<Entity, With<RemoteEntityProxy>>()
        .iter(&remote_world)
        .filter(|remote| !snapshot.rows.contains_key(remote))
        .collect();

    for remote in stale {
        let children: Vec<Entity> = remote_world
            .get::<Children>(remote)
            .map(|children| children.iter().copied().collect())
            .unwrap_or_default();
        for child in children {
            if let Ok(mut child) = remote_world.get_entity_mut(child) {
                child.remove::<ChildOf>();
            }
        }
        if let Ok(proxy) = remote_world.get_entity_mut(remote) {
            proxy.despawn();
        }
    }
}

fn spawn_missing(_world: &mut World, remote_world: &mut World, snapshot: &RemoteSnapshot) {
    for remote in &snapshot.order {
        let known = remote_world.get_entity(*remote).is_ok();
        if known {
            continue;
        }
        let _ = remote_world.spawn_at(*remote, RemoteEntityProxy {  });
    }
}

fn apply_hierarchy(_world: &mut World, remote_world: &mut World, snapshot: &RemoteSnapshot) {
    for remote in &snapshot.order {
        // let index = world.resource::<RemoteProxyIndex>();
        let proxy = remote;
        let parent = snapshot.rows[remote]
            .get(CHILD_OF)
            .and_then(|parent| serde_json::from_value::<Entity>(parent.clone()).ok())
            .and_then(|parent| Some(parent))
            .filter(|parent| *parent != *proxy);
        if remote_world.get::<ChildOf>(*proxy).map(ChildOf::parent) == parent {
            continue;
        }
        let Ok(mut proxy) = remote_world.get_entity_mut(*proxy) else {
            continue;
        };
        match parent {
            Some(parent) => {
                proxy.insert(ChildOf(parent));
            }
            None => {
                proxy.remove::<ChildOf>();
            }
        }
    }
}

fn apply_labels(world: &World, remote_world: &mut World, snapshot: &RemoteSnapshot) {
    let registry = world.resource::<AppTypeRegistry>().clone();
    let registry = registry.read();
    for remote in &snapshot.order {
        let proxy = *remote;
        let label = remote_label(world, remote_world, &registry, proxy, &snapshot.rows[remote])
            .unwrap_or_else(|| remote.to_string());
        if remote_world
            .get::<RemoteLabel>(proxy)
            .map(|label| label.0.as_str())
            != Some(label.as_str())
        {
            remote_world.entity_mut(proxy).insert(RemoteLabel(label));
        }
    }
}

/// Resolves the label of a remote entity the way the local tree does: its [`Name`], or else the
/// label-defining components it holds that are registered locally.
///
/// [`Name`]: bevy_ecs::name::Name
fn remote_label(
    world: &World,
    remote_world: &World,
    registry: &TypeRegistry,
    proxy: Entity,
    components: &Map<String, Value>,
) -> Option<String> {
    if let Some(name) = components.get(NAME).and_then(Value::as_str) {
        return Some(name.to_string());
    }
    let priorities = world.get_resource::<LabelResolutionRegistry>()?;
    let labels: Vec<(&str, _)> = components
        .keys()
        .filter_map(|type_path| {
            let registration = registry.get_with_type_path(type_path)?;
            let priority = priorities.get_priority_by_type_id(registration.type_id())?;
            Some((
                registration.type_info().type_path_table().short_path(),
                priority,
            ))
        })
        .collect();
    let data: Vec<ComponentLabelData> = labels
        .iter()
        .map(|(short_name, priority)| ComponentLabelData {
            component_id: ComponentId::new(0),
            short_name,
            label_definition_priority: Some(*priority),
        })
        .collect();
    resolve_label(&remote_world, proxy, &data).map(|label| label.label.as_str().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_ecs::name::Name;
    use bevy_reflect::TypeRegistryArc;
    use serde_json::json;

    pub(crate) fn test_world() -> World {
        let mut world = World::new();
        let registry = AppTypeRegistry(TypeRegistryArc::default());
        {
            let mut registry = registry.write();
            registry.register::<Name>();
            registry.register::<ChildOf>();
        }
        world.insert_resource(registry);
        world.insert_resource(InspectorSource::Remote(RemoteSource::localhost(15702)));
        world.init_resource::<InspectorSelection>();
        world.init_resource::<RemoteConnection>();
        // world.init_resource::<RemoteProxyIndex>();
        world.init_resource::<RemoteSnapshot>();
        world.init_resource::<EntityTreeSync>();
        world.init_resource::<DetailsPanelSync>();
        world
    }

    pub(crate) fn remote(index: u32) -> Entity {
        Entity::from_raw_u32(index).unwrap()
    }

    pub(crate) fn row(entity: Entity, components: Value) -> Value {
        json!({ "entity": entity, "components": components })
    }

    pub(crate) fn apply(world: &mut World, rows: Vec<Value>) {
        let rows: BrpQueryResponse = serde_json::from_value(Value::Array(rows)).unwrap();
        world.resource_mut::<RemoteSnapshot>().set(rows);
        apply_remote_snapshot(world);
    }

    #[test]
    fn entities_are_serialized_in_display_form() {
        let row = row(remote(5), json!({ CHILD_OF: remote(2) }));
        assert_eq!(row["entity"], json!("5v0"));
        assert_eq!(row["components"][CHILD_OF], json!("2v0"));
    }

    #[test]
    fn mirrors_remote_entities_as_disabled_proxies() {
        let mut world = test_world();
        let parent = remote(1);
        let child = remote(2);
        apply(
            &mut world,
            alloc::vec![
                row(parent, json!({ NAME: "Parent" })),
                row(child, json!({ NAME: "Child", CHILD_OF: parent })),
            ],
        );

        let parent_proxy = parent;
        let child_proxy = child;
        assert_ne!(parent_proxy, parent);
        assert!(world.get::<Disabled>(parent_proxy).is_some());
        assert!(world.get::<Name>(parent_proxy).is_none());
        assert_eq!(
            world.get::<ChildOf>(child_proxy).map(ChildOf::parent),
            Some(parent_proxy)
        );
        assert_eq!(proxy_label(&world, child_proxy).as_deref(), Some("Child"));
    }

    #[test]
    fn labels_unnamed_entities_with_their_remote_id() {
        let mut world = test_world();
        apply(
            &mut world,
            alloc::vec![row(remote(7), json!({ "demo::Unknown": 1 }))],
        );
        assert_eq!(
            proxy_label(&world, remote(7)).as_deref(),
            Some("7v0")
        );
    }

    #[test]
    fn follows_remote_despawns_and_reparenting() {
        let mut world = test_world();
        let a = remote(1);
        let b = remote(2);
        let c = remote(3);
        apply(
            &mut world,
            alloc::vec![
                row(a, json!({ NAME: "A" })),
                row(b, json!({ NAME: "B", CHILD_OF: a })),
                row(c, json!({ NAME: "C", CHILD_OF: b })),
            ],
        );
        let b_proxy = b;
        let c_proxy = c;

        apply(
            &mut world,
            alloc::vec![
                row(a, json!({ NAME: "A" })),
                row(c, json!({ NAME: "C", CHILD_OF: a })),
            ],
        );

        assert!(world.get_entity(b_proxy).is_err());
        assert!(world.get_entity(c_proxy).is_ok());
        assert_eq!(
            world.get::<ChildOf>(c_proxy).map(ChildOf::parent),
            Some(a)
        );
        // assert_eq!(world.resource::<RemoteProxyIndex>().len(), 2);
    }

    #[test]
    fn a_remote_id_colliding_with_a_local_entity_gets_its_own_proxy() {
        let mut world = test_world();
        let local = world.spawn(Name::new("Local")).id();
        apply(
            &mut world,
            alloc::vec![row(local, json!({ NAME: "Remote" }))],
        );

        let proxy = local;
        assert_ne!(proxy, local);
        assert_eq!(world.get::<Name>(local).map(Name::as_str), Some("Local"));
        assert_eq!(proxy_label(&world, proxy).as_deref(), Some("Remote"));
    }

    #[test]
    fn a_new_generation_replaces_the_proxy() {
        let mut world = test_world();
        let old = remote(4);
        let new =
            Entity::from_index_and_generation(old.index(), old.generation().after_versions(1));
        apply(&mut world, alloc::vec![row(old, json!({ NAME: "Old" }))]);
        let old_proxy = old;
        world.resource_mut::<InspectorSelection>().0 = Some(old_proxy);

        apply(&mut world, alloc::vec![row(new, json!({ NAME: "New" }))]);

        assert!(world.get_entity(old_proxy).is_err());
        assert_ne!(new, old_proxy);
    }

    #[test]
    fn switching_to_local_despawns_the_proxies() {
        let mut world = test_world();
        sync_remote_source(&mut world);
        apply(
            &mut world,
            alloc::vec![row(remote(1), json!({ NAME: "A" }))],
        );
        let proxy = remote(1);
        world.resource_mut::<InspectorSelection>().0 = Some(proxy);

        world.insert_resource(InspectorSource::Local);
        sync_remote_source(&mut world);

        assert!(world.get_entity(proxy).is_err());
        // assert!(world.resource::<RemoteProxyIndex>().is_empty());
        assert_eq!(world.resource::<InspectorSelection>().0, None);
        assert!(world.resource::<RemoteConnection>().client().is_none());
    }

    #[test]
    fn failed_requests_mark_the_connection_failed_and_retry_later() {
        let mut connection = RemoteConnection::default();
        let mut snapshot = RemoteSnapshot::default();
        let now = Duration::from_secs(3);
        finish_call(
            &mut connection,
            &mut snapshot,
            RequestKind::Query,
            Err(BrpClientError::InvalidResponse("refused".to_string())),
            now,
            Duration::ZERO,
        );
        assert!(matches!(connection.state, RemoteConnectionState::Failed(_)));
        assert_eq!(connection.next_poll, now + RETRY_INTERVAL);

        finish_call(
            &mut connection,
            &mut snapshot,
            RequestKind::Info,
            Ok(json!({ "app_name": "demo", "bevy_version": "0.20.0-dev", "sub_app": "main" })),
            now,
            Duration::ZERO,
        );
        assert_eq!(
            connection.state,
            RemoteConnectionState::Connected {
                app_name: "demo".to_string(),
                bevy_version: "0.20.0-dev".to_string(),
            }
        );
    }

    #[test]
    fn slow_answers_stretch_the_poll_interval() {
        let mut connection = RemoteConnection::default();
        let mut snapshot = RemoteSnapshot::default();
        let now = Duration::from_secs(10);
        finish_call(
            &mut connection,
            &mut snapshot,
            RequestKind::Query,
            Ok(json!([])),
            now,
            Duration::from_secs(1),
        );
        assert_eq!(connection.next_poll, now + Duration::from_secs(2));
    }
}
