/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, December 2023
 */

struct Uniforms {
    viewProjection: mat4x4f,
    lightDirection: vec3f,
    lightViewProjection: mat4x4f,
    shadowSize: f32,
    shadowBias: f32
};

struct Probe {
    position: vec3f,
    color: vec3f
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var colorTarget: texture_2d<f32>;
@group(0) @binding(2) var depthTarget: texture_depth_2d;
@group(0) @binding(3) var normalTarget: texture_2d<f32>;
@group(0) @binding(4) var positionTarget: texture_2d<f32>;
@group(0) @binding(5) var shadowTarget: texture_depth_2d;
@group(0) @binding(6) var shadowSampler: sampler_comparison;
@group(0) @binding(7) var<storage, read> probes: array<Probe>;

@vertex fn vs(@builtin(vertex_index) vertexIndex: u32) -> @builtin(position) vec4f {
    const positions: array<vec2f, 6> = array<vec2f, 6>(
        vec2f(-1, -1), vec2f(1, -1), vec2f(-1, 1),
        vec2f(-1, 1), vec2f(1, -1), vec2f(1, 1),
    );
    return vec4f(positions[vertexIndex], 0, 1);
} 

@fragment fn fs(@builtin(position) coord: vec4f) -> @location(0) vec4f {
    let uv: vec2i = vec2i(floor(coord.xy));
    let depth: f32 = textureLoad(depthTarget, uv, 0);
    if (depth >= 1) {
        discard;
    }
    let color: vec3f = textureLoad(colorTarget, uv, 0).rgb;
    let normal: vec3f = textureLoad(normalTarget, uv, 0).rgb;
    let position: vec3f = textureLoad(positionTarget, uv, 0).rgb;
    let theta: f32 = dot(normal, -uniforms.lightDirection);

    let positionLight: vec4f = uniforms.lightViewProjection * vec4(position, 1.0);
    let shadowPosition: vec3f = vec3f(positionLight.xy * vec2f(0.5, -0.5) + vec2f(0.5), positionLight.z);
    var visibility: f32 = 0;
    let invertShadowSize: f32 = 1 / uniforms.shadowSize;
    let bias: f32 = clamp(uniforms.shadowBias * tan(acos(theta)), 0, uniforms.shadowBias * 2);
    for (var y: f32 = -1; y <= 1; y += 1) {
        for (var x: f32 = -1; x <= 1; x += 1) {
            let xy: vec2f = shadowPosition.xy + vec2f(x, y) * invertShadowSize;
            visibility += textureSampleCompare(shadowTarget, shadowSampler, xy, shadowPosition.z - bias);
        }
    }
    visibility /= 9;
    if (shadowPosition.x <= 0 || shadowPosition.x >= 1 || 
        shadowPosition.y <= 0 || shadowPosition.y >= 1 || 
        shadowPosition.z <= 0 || shadowPosition.z >= 1) {
        visibility = 1;
    }


    let spread: f32 = 16;
    let base: vec3f = vec3f(0.5, 0.5, 0.5);
    let blend: vec3f = smoothstep(vec3f(0, 0, 0), vec3f(1, 1, 1), fract(position / spread));

    var bottomBackLeft: vec3f = base;
    var bottomBackRight: vec3f = base;
    var bottomFrontLeft: vec3f = base;
    var bottomFrontRight: vec3f = base;
    var topBackLeft: vec3f = base;
    var topBackRight: vec3f = base;
    var topFrontLeft: vec3f = base;
    var topFrontRight: vec3f = base;

    for (var i: u32 = 0; i < 75; i++) {
        let probe: Probe = probes[i];
        let diff: vec3f = probe.position - position;

        if ((diff.x > -spread && diff.x <= 0) && (diff.y > -spread && diff.y <= 0) && (diff.z > -spread && diff.z <= 0)) {
            bottomBackLeft = probe.color;
        } else if ((diff.x < spread && diff.x >= 0) && (diff.y > -spread && diff.y <= 0) && (diff.z > -spread && diff.z <= 0)) {
            bottomBackRight = probe.color;
        } else if ((diff.x > -spread && diff.x <= 0) && (diff.y > -spread && diff.y <= 0) && (diff.z < spread && diff.z >= 0)) {
            bottomFrontLeft = probe.color;
        } else if ((diff.x < spread && diff.x >= 0) && (diff.y > -spread && diff.y <= 0) && (diff.z < spread && diff.z >= 0)) {
            bottomFrontRight = probe.color;
        } else if ((diff.x > -spread && diff.x <= 0) && (diff.y < spread && diff.y >= 0) && (diff.z > -spread && diff.z <= 0)) {
            topBackLeft = probe.color;
        } else if ((diff.x < spread && diff.x >= 0) && (diff.y < spread && diff.y >= 0) && (diff.z > -spread && diff.z <= 0)) {
            topBackRight = probe.color;
        } else if ((diff.x > -spread && diff.x <= 0) && (diff.y < spread && diff.y >= 0) && (diff.z < spread && diff.z >= 0)) {
            topFrontLeft = probe.color;
        } else if ((diff.x < spread && diff.x >= 0) && (diff.y < spread && diff.y >= 0) && (diff.z < spread && diff.z >= 0)) {
            topFrontRight = probe.color;
        }
    }
    
    let bottomBack: vec3f = mix(bottomBackLeft, bottomBackRight, blend.x);
    let bottomFront: vec3f = mix(bottomFrontLeft, bottomFrontRight, blend.x);
    let topBack: vec3f = mix(topBackLeft, topBackRight, blend.x);
    let topFront: vec3f = mix(topFrontLeft, topFrontRight, blend.x);

    let bottom: vec3f = mix(bottomBack, bottomFront, blend.z);
    let top: vec3f = mix(topBack, topFront, blend.z);

    let gi: vec3f = mix(bottom, top, blend.y);
    
    let lambert: f32 = max(0, theta);
    let halfLambert: f32 = pow(lambert * 0.5 + 0.5, 1);

    let shade: f32 = lambert * visibility;
    let inverseShade: f32 = 1 - shade;

    let direct: vec3f = color * shade;
    let indirect: vec3f = gi * inverseShade * 0.35;

    return vec4f(direct + indirect, 1);
}