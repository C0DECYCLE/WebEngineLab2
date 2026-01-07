/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger
 */

#include structs.wgsl;
#include depthColorPack.wgsl;

override SCREEN_WIDTH: u32;
override SCREEN_HEIGHT: u32;

@group(0) @binding(0) var<storage, read> frame: array<u32>;

@vertex fn vs(
    @builtin(vertex_index) vertexIndex: u32, 
) -> @builtin(position) vec4f {
    let uv: vec2u = vec2u((vertexIndex << 1) & 2, vertexIndex & 2);
    return vec4f(vec2f(uv) * 2 - 1, 0, 1);
}

@fragment fn fs(
    @builtin(position) position: vec4f
) -> @location(0) vec4f {
    let pixel: u32 = u32(position.y) * SCREEN_WIDTH + u32(position.x);
    let value: u32 = frame[pixel];
    if (value == 0) {
        discard;
    }
    let color: vec3f = unpack_color(value);
    //let depth: f32 = unpack_depth(value);
    return vec4f(color, 1);
}