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

@vertex fn vs(@builtin(vertex_index) vertexIndex: u32, @builtin(instance_index) instanceIndex: u32) -> VertexShaderOut {
    let vertex: Vertex = vertecies[vertexIndex];
    let instance: Instance = instances[instanceIndex];

    var worldPosition: vec3f = vertex.position;
    worldPosition *= instance.size;
    worldPosition += instance.position;
    worldPosition.y += noise3(worldPosition * 0.1) * 10;
    
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

fn mod289(x: vec4f) -> vec4f {
    return x - floor(x * (1. / 289.)) * 289.;
}

fn perm4(x: vec4f) -> vec4f {
    return mod289(((x * 34.) + 1.) * x);
}

fn noise3(p: vec3f) -> f32 {
    let a: vec3f = floor(p);
    var d: vec3f = p - a;
    d = d * d * (3. - 2. * d);
    let b: vec4f = a.xxyy + vec4f(0., 1., 0., 1.);
    let k1: vec4f = perm4(b.xyxy);
    let k2: vec4f = perm4(k1.xyxy + b.zzww);
    let c: vec4f = k2 + a.zzzz;
    let k3: vec4f = perm4(c);
    let k4: vec4f = perm4(c + 1.);
    let o1: vec4f = fract(k3 * (1. / 41.));
    let o2: vec4f = fract(k4 * (1. / 41.));
    let o3: vec4f = o2 * d.z + o1 * (1. - d.z);
    let o4: vec2f = o3.yw * d.x + o3.xz * (1. - d.x);
    return o4.y * d.y + o4.x * (1. - d.y);
}