/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, January 2026
 */

const COLOR_BITS: u32 = 5u + 6u + 5u;
const DEPTH_BITS: u32 = 32u - COLOR_BITS;

fn pack_depth(depth: f32) -> u32 {
    let d: f32 = 1 - clamp(depth, 0, 1);
    let max_depth: u32 = (1u << DEPTH_BITS) - 1u;
    return u32(d * f32(max_depth));
}

fn pack_color(color: vec3f) -> u32 {
    let r_bits: u32 = 5u;
    let g_bits: u32 = 6u;
    let b_bits: u32 = 5u;
    let r_max: u32 = (1u << r_bits) - 1u;
    let g_max: u32 = (1u << g_bits) - 1u;
    let b_max: u32 = (1u << b_bits) - 1u;
    let r: u32 = u32(clamp(color.r, 0, 1) * f32(r_max));
    let g: u32 = u32(clamp(color.g, 0, 1) * f32(g_max));
    let b: u32 = u32(clamp(color.b, 0, 1) * f32(b_max));
    return (r << (g_bits + b_bits)) | (g << b_bits) | b;
}

fn pack_depth_color(depth: f32, color: vec3f) -> u32 {
    let d: u32 = pack_depth(depth);
    let c: u32 = pack_color(color);
    return (d << COLOR_BITS) | c;
}

fn unpack_depth(value: u32) -> f32 {
    let max_depth: u32 = (1u << DEPTH_BITS) - 1u;
    return f32(value >> COLOR_BITS) / f32(max_depth);
}

fn unpack_color(value: u32) -> vec3f {
    let r_bits: u32 = 5u;
    let g_bits: u32 = 6u;
    let b_bits: u32 = 5u;
    let r_max: u32 = (1u << r_bits) - 1u;
    let g_max: u32 = (1u << g_bits) - 1u;
    let b_max: u32 = (1u << b_bits) - 1u;
    let c: u32 = value & ((1u << COLOR_BITS) - 1u);
    let r: u32 = (c >> (g_bits + b_bits)) & r_max;
    let g: u32 = (c >> b_bits) & g_max;
    let b: u32 =  c & b_max;
    return vec3f(
        f32(r) / f32(r_max),
        f32(g) / f32(g_max),
        f32(b) / f32(b_max),
    );
}