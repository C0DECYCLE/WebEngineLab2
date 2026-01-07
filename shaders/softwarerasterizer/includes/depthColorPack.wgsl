/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger
 */

const DEPTH_BITS: u32 = 20;
const COLOR_BITS: u32 = 12;

const DEPTH_MAX: f32 = 1048575;

fn pack_depth(depth: f32) -> u32 {
    let d: f32 = 1 - clamp(depth, 0, 1);
    return u32(d * DEPTH_MAX);
}

fn pack_color(color: vec3f) -> u32 {
    let r = u32(clamp(color.r, 0, 1) * 15);
    let g = u32(clamp(color.g, 0, 1) * 15);
    let b = u32(clamp(color.b, 0, 1) * 15);
    return (r << 8) | (g << 4) | b;
}

fn pack_depth_color(depth: f32, color: vec3f) -> u32 {
    let d: u32 = pack_depth(depth);
    let c: u32 = pack_color(color);
    return (d << COLOR_BITS) | c;
}

fn unpack_depth(value: u32) -> f32 {
    return f32(value >> COLOR_BITS) / DEPTH_MAX;
}

fn unpack_color(value: u32) -> vec3f {
    let c: u32 = value & 0xFFF;
    let r: f32 = f32((c >> 8) & 0xF) / 15;
    let g: f32 = f32((c >> 4) & 0xF) / 15;
    let b: f32 = f32(c & 0xF) / 15;
    return vec3f(r, g, b);
}