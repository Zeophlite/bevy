---
title: "`SubApp` renamed to `App`"
pull_requests: []
---

- `SubApp` renamed to `App`
- `App` renamed to `Bevy`
- Removed `add_system`, etc. from `Bevy` (which just forwarded to the main `App`)

```rs
// before
impl Plugin for Foo<A> {
    fn build(&self, app: &mut App) {
        // do thing with app

        // optional:
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            // do thing with render_app
        }
    }
}

// after
impl Plugin for Foo<A> {
    fn build(&self, bevy: &mut Bevy) {
        let app = bevy.main_mut();
        // do thing with app

        // optional:
        if let Some(render_app) = bevy.get_app_mut(RenderApp) {
            // do thing with render_app
        }
    }
}
```

- if you need both, use `let (app, maybe_render_app) = bevy.get_main_and_app_mut(RenderApp);`

- before: `Plugin` only operated on `App` against the main `SubApp` , but plugin infrastructure was stored on all `SubApp`'s


```rs
// before
fn main() {
    App::new()
        .add_systems(Startup, setup_mesh_and_animation)
        .run();
}

// after
fn main() {
    Bevy::new()
        .main_mut() // explicitly move to Main subapp
        .add_systems(Startup, setup_mesh_and_animation)
        .run();
}
```