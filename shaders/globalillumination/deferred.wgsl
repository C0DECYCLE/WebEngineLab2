/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, December 2023
 */

@group(0) @binding(0) var colorTarget: texture_2d<f32>;
@group(0) @binding(1) var depthTarget: texture_depth_2d;
@group(0) @binding(2) var normalTarget: texture_2d<f32>;

@vertex fn vs(@builtin(vertex_index) vertexIndex: u32) -> @builtin(position) vec4f {
    const positions: array<vec2f, 6> = array<vec2f, 6>(
        vec2f(-1, -1), vec2f(1, -1), vec2f(-1, 1),
        vec2f(-1, 1), vec2f(1, -1), vec2f(1, 1),
    );
    return vec4f(positions[vertexIndex], 0, 1);
}

@fragment fn fs(@builtin(position) coord: vec4f) -> @location(0) vec4f {
    const direction: vec3f = normalize(vec3f(1, 1, 1));
    let uv: vec2i = vec2i(floor(coord.xy));
    let depth: f32 = textureLoad(depthTarget, uv, 0);
    if (depth >= 1) {
        discard;
    }
    let color: vec3f = textureLoad(colorTarget, uv, 0).rgb;
    let normal: vec3f = textureLoad(normalTarget, uv, 0).rgb;
    let magnitude: f32 = max(0, dot(normal, -direction));
    let half: f32 = pow(magnitude * 0.5 + 0.5, 2);
    return vec4f(color * half, 1);
}
