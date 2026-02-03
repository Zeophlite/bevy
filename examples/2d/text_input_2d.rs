//! basic bevy 2d text input
use std::any::TypeId;

use bevy::{
    camera::{primitives::Aabb, visibility::VisibilityClass},
    color::palettes::{
        css::{GREEN, NAVY, RED, YELLOW},
        tailwind::{BLUE_900, GRAY_300, GRAY_400, SKY_300},
    },
    input::keyboard::{Key, KeyboardInput},
    prelude::*,
    render::{sync_world::TemporaryRenderEntity, Extract, RenderApp},
    sprite::Anchor,
    sprite_render::{
        ExtractedSlice, ExtractedSlices, ExtractedSprite, ExtractedSpriteKind, ExtractedSprites,
    },
    text::{
        CursorBlink, LineBreak, Motion, Placeholder, PlaceholderLayout, PositionedGlyph,
        TextBounds, TextCursorBlinkInterval, TextEdit, TextEdits, TextInputBuffer,
        TextInputSystems, TextInputTarget, TextLayoutInfo,
    },
    window::PrimaryWindow,
};

use crate::text_input::{
    extract_text_input, update_inputs, update_targets, DisplayConfig, Overwrite,
    TextInputKeyConfig, TextInputSize, TextSubmission,
};

#[path = "../helpers/text_input.rs"]
mod text_input;

fn main() {
    let mut app = App::new();

    app.add_plugins(DefaultPlugins)
        .add_systems(Startup, setup)
        .insert_resource(TextInputKeyConfig {
            allow_indent: true,
            allow_newline: true,
            allow_scroll: true,
        })
        .add_systems(
            PostUpdate,
            // TODO: these are both from helpers
            (update_inputs, update_targets).before(TextInputSystems),
        )
        .add_message::<TextSubmission>()
        .add_systems(Update, handle_submissions);
    app.sub_app_mut(RenderApp)
        .add_systems(ExtractSchedule, extract_text_input);

    app.run();
}

fn setup(mut commands: Commands) {
    commands.spawn(Camera2d);

    commands.spawn((
        Text2d::new("submit with SHIFT + ENTER"),
        Anchor::TOP_CENTER,
        TextBounds {
            width: Some(500.),
            height: None,
        },
        TextLayout {
            linebreak: LineBreak::AnyCharacter,
            justify: Justify::Left,
        },
        Transform::from_translation(Vec3::new(0., -25., 0.)),
    ));

    commands.spawn((
        TextInputBuffer {
            ..Default::default()
        },
        CursorBlink {
            cursor_blink_timer: 0.,
        },
        Overwrite::default(),
        TextInputSize(Vec2::new(500., 250.)),
        Transform::from_translation(Vec3::new(0., 150., 0.)), // This is in UI coordinates
        Placeholder::new("type here.."),
        Visibility::default(),
        VisibilityClass([TypeId::of::<Sprite>()].into()), // Text input is rendered as sprites
        Anchor::CENTER,
        DisplayConfig {
            placeholder_text: SKY_300.into(),
            input_text: GRAY_300.into(),
            input_background: NAVY.into(),
            selected_highlight: BLUE_900.into(),
            cursor: GRAY_400.into(),
        },
    ));
}

fn handle_submissions(
    mut submit_events: MessageReader<TextSubmission>,
    text_query: Single<&mut Text2d>,
    buffer_query: Single<&TextInputBuffer>,
) {
    let mut text = text_query.into_inner();
    let buffer = *buffer_query;

    for _ in submit_events.read() {
        text.0 = buffer.get_text().clone();
    }
}
