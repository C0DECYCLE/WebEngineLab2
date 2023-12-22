/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, December 2023
 */

@group(0) @binding(0) var colorTarget: texture_2d<f32>;
@group(0) @binding(1) var depthTarget: texture_depth_2d;

@vertex fn vs(@builtin(vertex_index) vertexIndex: u32) -> @builtin(position) vec4f {
    const positions: array<vec2f, 6> = array<vec2f, 6>(
        vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
        vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
    );
    return vec4f(positions[vertexIndex], 0.0, 1.0);
}

@fragment fn fs(@builtin(position) coord: vec4f) -> @location(0) vec4f {
    let uv: vec2i = vec2i(floor(coord.xy));
    let depth: f32 = textureLoad(depthTarget, uv, 0);
    if (depth >= 1.0) {
        discard;
    }
    let color: vec4f = textureLoad(colorTarget, uv, 0);
    return color;
}
