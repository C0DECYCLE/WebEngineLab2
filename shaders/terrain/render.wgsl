/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, October 2023
 */

struct Uniforms {
    viewProjection: mat4x4f,
};

struct Vertex {
    position: vec3f,
};

struct Instance {
    position: vec3f,
    size: f32
};

struct VertexShaderOut {
    @builtin(position) position: vec4f,
    @location(0) worldPosition: vec3f
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> vertecies: array<Vertex>;
@group(0) @binding(2) var<storage, read> instances: array<Instance>;
@group(0) @binding(3) var heightmap: texture_2d<f32>;

@vertex fn vs(@builtin(vertex_index) vertexIndex: u32, @builtin(instance_index) instanceIndex: u32) -> VertexShaderOut {
    let vertex: Vertex = vertecies[vertexIndex];
    let instance: Instance = instances[instanceIndex];

    var worldPosition: vec3f = vertex.position;
    worldPosition *= instance.size;
    worldPosition += instance.position;
    
    //worldPosition.y += fbm(worldPosition.xz * 0.01) * 10 + length(worldPosition.xz * 0.01) * 10;
    let uv: vec2u = vec2u(((worldPosition.xz / 256) + 0.5) * 4096) + 1;
    let texel: vec4f = textureLoad(heightmap, uv, 0); 
    worldPosition.y += ((texel.x + texel.y + texel.z) / 3) * 100;

    var out: VertexShaderOut;
    out.position = uniforms.viewProjection * vec4f(worldPosition, 1);
    out.worldPosition = worldPosition;
    return out;
}

@fragment fn fs(in: VertexShaderOut) -> @location(0) vec4f {
    var position: vec3f = in.worldPosition;
    let normal: vec3f = normalize(cross(dpdx(position), dpdy(position)));
    return vec4f(normal * 0.5 + 0.5, 1.0);
}

//  MIT License. © Ian McEwan, Stefan Gustavson, Munrocket, Johan Helsing
fn mod289(x: vec2f) -> vec2f {
    return x - floor(x * (1. / 289.)) * 289.;
}
fn mod289_3(x: vec3f) -> vec3f {
    return x - floor(x * (1. / 289.)) * 289.;
}
fn permute3(x: vec3f) -> vec3f {
    return mod289_3(((x * 34.) + 1.) * x);
}
fn simplexNoise2(v: vec2f) -> f32 {
    let C = vec4(
        0.211324865405187,
        0.366025403784439,
        -0.577350269189626,
        0.024390243902439
    );
    var i = floor(v + dot(v, C.yy));
    let x0 = v - i + dot(i, C.xx);
    var i1 = select(vec2(0., 1.), vec2(1., 0.), x0.x > x0.y);
    var x12 = x0.xyxy + C.xxzz;
    x12.x = x12.x - i1.x;
    x12.y = x12.y - i1.y;
    i = mod289(i);
    var p = permute3(permute3(i.y + vec3(0., i1.y, 1.)) + i.x + vec3(0., i1.x, 1.));
    var m = max(0.5 - vec3(dot(x0, x0), dot(x12.xy, x12.xy), dot(x12.zw, x12.zw)), vec3(0.));
    m *= m;
    m *= m;
    let x = 2. * fract(p * C.www) - 1.;
    let h = abs(x) - 0.5;
    let ox = floor(x + 0.5);
    let a0 = x - ox;
    m *= 1.79284291400159 - 0.85373472095314 * (a0 * a0 + h * h);
    let g = vec3(a0.x * x0.x + h.x * x0.y, a0.yz * x12.xz + h.yz * x12.yw);
    return 130. * dot(m, g);
}

//  MIT License. © Inigo Quilez, Munrocket
const m2: mat2x2f = mat2x2f(vec2f(0.8, 0.6), vec2f(-0.6, 0.8));
fn fbm(p: vec2f) -> f32 {
    var mp = p;
    var f: f32 = 0.;
    f = f + 0.5000 * simplexNoise2(mp); mp = m2 * mp * 2.02;
    f = f + 0.2500 * simplexNoise2(mp); mp = m2 * mp * 2.03;
    f = f + 0.1250 * simplexNoise2(mp); mp = m2 * mp * 2.01;
    f = f + 0.0625 * simplexNoise2(mp);
    return f / 0.9375;
}