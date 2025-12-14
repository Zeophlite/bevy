//! Shows text rendering with moving, rotating and scaling text.
//!
//! Note that this uses [`Text2d`] to display text alongside your other entities in a 2D scene.
//!
//! For an example on how to render text as part of a user interface, independent from the world
//! viewport, you may want to look at `games/contributors.rs` or `ui/text.rs`.

use bevy::{
    camera::primitives::Aabb,
    color::palettes::{
        css::{GREEN, NAVY, RED, YELLOW, *},
        tailwind::{BLUE_900, GRAY_300, GRAY_400, SKY_300},
    },
    input::keyboard::{Key, KeyboardInput},
    math::ops,
    prelude::*,
    sprite::{Anchor, Text2dShadow},
    text::{
        FontSmoothing, LineBreak, Text2dUpdateSystems, TextBounds, TextInputAttributes,
        TextInputBuffer, TextInputSystems,
    },
};
use bevy_render::RenderApp;

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
            allow_indent: false,
            allow_newline: false,
            allow_scroll: false,
        })
        .insert_resource(SelectedTextable { idx: -1 })
        .add_systems(PreUpdate, rotate_selected_input)
        .add_systems(
            PostUpdate,
            // TODO: these are both from helpers
            // TODO: allow disable scroll in inputs
            (update_inputs, update_targets).before(TextInputSystems),
        )
        .add_message::<TextSubmission>()
        .add_systems(Update, handle_submissions)
        .add_systems(
            Update,
            (animate_translation, animate_rotation, animate_scale),
        );
    app.sub_app_mut(RenderApp)
        .add_systems(ExtractSchedule, extract_text_input);

    app.run();
}

#[derive(Component)]
struct Textable(i32);

#[derive(Resource)]
struct SelectedTextable {
    pub idx: i32,
}

#[derive(Component)]
struct AnimateTranslation;

#[derive(Component)]
struct AnimateRotation;

#[derive(Component)]
struct AnimateScale;

fn setup(mut commands: Commands, asset_server: Res<AssetServer>) {
    let font = asset_server.load("fonts/FiraSans-Bold.ttf");
    let text_font = TextFont {
        font: font.clone(),
        font_size: 50.0,
        ..default()
    };
    let text_justification = Justify::Center;
    commands.spawn(Camera2d);

    let small_text_font = TextFont {
        font: font.clone(),
        font_size: 20.0,
        ..default()
    };
    commands.spawn((
        Text2d::new("Press Tab to cycle input, shift tab to cycle backwards"),
        small_text_font.clone(),
        TextBackgroundColor(Color::BLACK.with_alpha(0.5)),
        Text2dShadow::default(),
        Anchor::TOP_LEFT,
        Transform::from_translation(Vec3::new(-600.0, 340.0, 0.0)),
        TextInputSize(Vec2::new(100., 40.)),
    ));

    // Demonstrate changing translation
    commands.spawn((
        Text2d::new(" translation "),
        text_font.clone(),
        TextLayout::new_with_justify(text_justification),
        TextBackgroundColor(Color::BLACK.with_alpha(0.5)),
        Text2dShadow::default(),
        Textable(0),
        Anchor::TOP_LEFT,
        TextInputSize(Vec2::new(100., 40.)),
        DisplayConfig {
            placeholder_text: SKY_300.into(),
            input_text: GRAY_300.into(),
            input_background: NAVY.into(),
            selected_highlight: BLUE_900.into(),
            cursor: GRAY_400.into(),
        },
        AnimateTranslation,
    ));

    // Demonstrate changing rotation
    commands.spawn((
        Text2d::new(" rotation "),
        text_font.clone(),
        TextLayout::new_with_justify(text_justification),
        TextBackgroundColor(Color::BLACK.with_alpha(0.5)),
        Text2dShadow::default(),
        Textable(1),
        Anchor::TOP_LEFT,
        TextInputSize(Vec2::new(100., 40.)),
        AnimateRotation,
    ));

    // Demonstrate changing scale
    commands.spawn((
        Text2d::new(" scale "),
        text_font,
        TextLayout::new_with_justify(text_justification),
        Transform::from_translation(Vec3::new(400.0, 0.0, 0.0)),
        TextBackgroundColor(Color::BLACK.with_alpha(0.5)),
        Text2dShadow::default(),
        // Textable(2),
        Anchor::TOP_LEFT,
        TextInputSize(Vec2::new(100., 40.)),
        AnimateScale,
    ));

    // Demonstrate text wrapping
    let slightly_smaller_text_font = TextFont {
        font,
        font_size: 35.0,
        ..default()
    };
    let box_size = Vec2::new(300.0, 200.0);
    let box_position = Vec2::new(0.0, -250.0);
    let box_color = Color::srgb(0.25, 0.25, 0.55);
    let text_shadow_color = box_color.darker(0.05);
    commands.spawn((
        Sprite::from_color(Color::srgb(0.25, 0.25, 0.55), box_size),
        Transform::from_translation(box_position.extend(0.0)),
        children![(
            Text2d::new("this text wraps in the box\n(Unicode linebreaks)"),
            slightly_smaller_text_font.clone(),
            TextLayout::new(Justify::Left, LineBreak::WordBoundary),
            // Wrap text in the rectangle
            TextBounds::from(box_size),
            // Ensure the text is drawn on top of the box
            Transform::from_translation(Vec3::Z),
            // Add a shadow to the text
            Text2dShadow {
                color: text_shadow_color,
                ..default()
            },
            Underline,
            // Textable(3),
            Anchor::TOP_LEFT,
            TextInputSize(Vec2::new(100., 40.)),
            // observer(over_text),
            // observer(out_text),
        )],
    ));

    let other_box_size = Vec2::new(300.0, 200.0);
    let other_box_position = Vec2::new(320.0, -250.0);
    commands.spawn((
        Sprite::from_color(Color::srgb(0.25, 0.25, 0.55), other_box_size),
        Transform::from_translation(other_box_position.extend(0.0)),
        children![(
            Text2d::new("this text wraps in the box\n(AnyCharacter linebreaks)"),
            slightly_smaller_text_font.clone(),
            TextLayout::new(Justify::Left, LineBreak::AnyCharacter),
            // Wrap text in the rectangle
            TextBounds::from(other_box_size),
            // Ensure the text is drawn on top of the box
            Transform::from_translation(Vec3::Z),
            // Add a shadow to the text
            Text2dShadow {
                color: text_shadow_color,
                ..default()
            },
            // Textable(4),
            Anchor::TOP_LEFT,
            TextInputSize(Vec2::new(100., 40.)),
            // observe(over_text),
            // observe(out_text),
        )],
    ));

    // Demonstrate font smoothing off
    commands.spawn((
        Text2d::new("This text has\nFontSmoothing::None\nAnd Justify::Center"),
        slightly_smaller_text_font
            .clone()
            .with_font_smoothing(FontSmoothing::None),
        TextLayout::new_with_justify(Justify::Center),
        Transform::from_translation(Vec3::new(-400.0, -250.0, 0.0)),
        // Add a black shadow to the text
        Text2dShadow::default(),
        // Textable(5),
        Anchor::TOP_LEFT,
        TextInputSize(Vec2::new(100., 40.)),
    ));

    let make_child = move |(text_anchor, color, delta): (Anchor, Color, i32)| {
        (
            Text2d::new(" Anchor".to_string()),
            slightly_smaller_text_font.clone(),
            text_anchor,
            TextBackgroundColor(Color::WHITE.darker(0.8)),
            Transform::from_translation(-1. * Vec3::Z),
            // Textable(6 + delta * 3),
            TextInputSize(Vec2::new(100., 40.)),
            children![
                (
                    TextSpan("::".to_string()),
                    slightly_smaller_text_font.clone(),
                    TextColor(LIGHT_GREY.into()),
                    TextBackgroundColor(DARK_BLUE.into()),
                    // Textable(7 + delta * 3),
                    Anchor::TOP_LEFT,
                    TextInputSize(Vec2::new(100., 40.)),
                ),
                (
                    TextSpan(format!("{text_anchor:?} ")),
                    slightly_smaller_text_font.clone(),
                    TextColor(color),
                    TextBackgroundColor(color.darker(0.3)),
                    // Textable(8 + delta * 3),
                    Anchor::TOP_LEFT,
                    TextInputSize(Vec2::new(100., 40.)),
                )
            ],
        )
    };

    commands.spawn((
        Sprite {
            color: Color::Srgba(LIGHT_CYAN),
            custom_size: Some(Vec2::new(10., 10.)),
            ..Default::default()
        },
        Transform::from_translation(250. * Vec3::Y),
        children![
            make_child((Anchor::TOP_LEFT, Color::Srgba(LIGHT_SALMON), 0)),
            make_child((Anchor::TOP_RIGHT, Color::Srgba(LIGHT_GREEN), 1)),
            make_child((Anchor::BOTTOM_RIGHT, Color::Srgba(LIGHT_BLUE), 2)),
            make_child((Anchor::BOTTOM_LEFT, Color::Srgba(LIGHT_YELLOW), 3)),
        ],
    ));
}

fn animate_translation(
    time: Res<Time>,
    mut query: Query<&mut Transform, (With<Text2d>, With<AnimateTranslation>)>,
) {
    for mut transform in &mut query {
        transform.translation.x = 100.0 * ops::sin(time.elapsed_secs()) - 400.0;
        transform.translation.y = 100.0 * ops::cos(time.elapsed_secs());
    }
}

fn animate_rotation(
    time: Res<Time>,
    mut query: Query<&mut Transform, (With<Text2d>, With<AnimateRotation>)>,
) {
    for mut transform in &mut query {
        transform.rotation = Quat::from_rotation_z(ops::cos(time.elapsed_secs()));
    }
}

fn animate_scale(
    time: Res<Time>,
    mut query: Query<&mut Transform, (With<Text2d>, With<AnimateScale>)>,
) {
    // Consider changing font-size instead of scaling the transform. Scaling a Text2D will scale the
    // rendered quad, resulting in a pixellated look.
    for mut transform in &mut query {
        let scale = (ops::sin(time.elapsed_secs()) + 1.1) * 2.0;
        transform.scale.x = scale;
        transform.scale.y = scale;
    }
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

fn rotate_selected_input(
    mut commands: Commands,
    mut keyboard_events: MessageReader<KeyboardInput>,
    keyboard_state: Res<ButtonInput<Key>>,
    mut selectable_textable: ResMut<SelectedTextable>,
    mut query: Query<(
        Entity,
        &mut Textable,
        &mut Text2d,
        Option<&mut TextInputBuffer>,
    )>,
) {
    let current_selection = selectable_textable.idx;
    let mut delta: i32 = 0;

    for key_input in keyboard_events.read() {
        if !key_input.state.is_pressed() {
            continue;
        }

        let pressed_key = &key_input.logical_key;
        let is_shift_pressed = keyboard_state.pressed(Key::Shift);

        match &pressed_key {
            Key::Tab => {
                if is_shift_pressed {
                    // backward
                    println!("go backward");
                    delta -= 1;
                } else {
                    // forward
                    println!("go forward");
                    delta += 1;
                }
            }
            _ => {}
        }
    }

    if delta == 0 {
        return;
    }

    println!("current_selection is {:?}", current_selection);
    println!("delta is {:?}", delta);

    let count_textable: i32 = query.iter().len() as i32;

    let mut new_idx = current_selection + delta;
    if new_idx >= count_textable {
        new_idx = 0;
    }

    selectable_textable.idx = new_idx;

    for (entity, mut textable, mut text, maybe_buffer) in query.iter_mut() {
        println!(
            "found textable {:?} {:?} #{:?}#",
            textable.0,
            maybe_buffer.is_some(),
            text.0
        );

        if textable.0 == new_idx {
            let mut b = TextInputBuffer::default();
            b.with_text(&text.0);
            // let c = TextInputAttributes::from(value);
            commands.entity(entity).insert(b);
        } else {
            if let Some(buffer) = maybe_buffer {
                // remove buffer
            }
            commands.entity(entity).remove::<TextInputBuffer>();
        }
    }
}
