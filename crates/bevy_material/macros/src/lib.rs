#![expect(missing_docs, reason = "Not all docs are written yet, see #3492.")]
#![cfg_attr(docsrs, feature(doc_auto_cfg))]

mod specializer;

use bevy_macro_utils::{derive_label, BevyManifest};
use proc_macro::TokenStream;
use quote::format_ident;
use syn::{parse_macro_input, DeriveInput};

pub(crate) fn bevy_material_path() -> syn::Path {
    BevyManifest::shared().get_path("crate")
}

pub(crate) fn bevy_ecs_path() -> syn::Path {
    BevyManifest::shared().get_path("bevy_ecs")
}

/// Derive macro generating an impl of the trait `Specializer`
///
/// This only works for structs whose members all implement `Specializer`
#[proc_macro_derive(Specializer, attributes(specialize, key, base_descriptor))]
pub fn derive_specialize(input: TokenStream) -> TokenStream {
    specializer::impl_specializer(input)
}

/// Derive macro generating the most common impl of the trait `SpecializerKey`
#[proc_macro_derive(SpecializerKey)]
pub fn derive_specializer_key(input: TokenStream) -> TokenStream {
    specializer::impl_specializer_key(input)
}

#[proc_macro_derive(ShaderLabel)]
pub fn derive_shader_label(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let mut trait_path = bevy_material_path();
    trait_path
        .segments
        .push(format_ident!("render_phase").into());
    trait_path
        .segments
        .push(format_ident!("ShaderLabel").into());
    derive_label(input, "ShaderLabel", &trait_path)
}

#[proc_macro_derive(DrawFunctionLabel)]
pub fn derive_draw_function_label(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let mut trait_path = bevy_material_path();
    trait_path
        .segments
        .push(format_ident!("render_phase").into());
    trait_path
        .segments
        .push(format_ident!("DrawFunctionLabel").into());
    derive_label(input, "DrawFunctionLabel", &trait_path)
}
