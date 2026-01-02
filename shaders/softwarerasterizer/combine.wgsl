/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, January 2026
 */

override SCREEN_WIDTH: u32;
override SCREEN_HEIGHT: u32;

@group(0) @binding(0) var<storage, read> frame: array<u32>;

@vertex fn vs(
    @builtin(vertex_index) vertexIndex: u32, 
) -> @builtin(position) vec4f {
    let positions: array<vec2f, 3> = array<vec2f, 3>(
        vec2f(-1, -3),
        vec2f(3, 1),
        vec2f(-1, 1)
    );
    return vec4f(positions[vertexIndex], 0, 1);
}

@fragment fn fs(
    @builtin(position) clipspace: vec4f
) -> @location(0) vec4f {
    let index: u32 = u32(clipspace.y) * SCREEN_WIDTH + u32(clipspace.x);
    let value: u32 = frame[index];
    if (value == 0) {
        discard;
    }
    let color: vec3f = unpack_color(value);
    //let depth: f32 = unpack_depth(value);
    return vec4f(color, 1);
}

fn unpack_depth(value: u32) -> f32 {
    return 1 - (f32(value >> 8) / 16777215);
}

fn unpack_color(value: u32) -> vec3f {
    let c: u32 = value & 0xFFu;
    let r: f32 = f32((c >> 5) & 7) / 7;
    let g: f32 = f32((c >> 2) & 7) / 7;
    let b: f32 = f32(c & 3) / 3;
    return vec3f(r, g, b);
}