/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, January 2026
 */

#include structs.wgsl;
#include depthColorPack.wgsl;

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