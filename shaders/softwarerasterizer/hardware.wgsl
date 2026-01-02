/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, January 2026
 */

override SCREEN_WIDTH: u32;
override SCREEN_HEIGHT: u32;

struct Uniforms {
    viewProjection: mat4x4f,
};

struct Instance {
    position: vec3f,
};

struct Vertex {
    position: vec3f,
};

struct VertexShaderOut {
    @builtin(position) clipspace: vec4f,
    @location(0) worldspace: vec3f,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> vertices: array<Vertex>;
@group(0) @binding(2) var<storage, read> instances: array<Instance>;
@group(0) @binding(3) var<storage, read_write> frame: array<atomic<u32>>;

@vertex fn vs(
    @builtin(vertex_index) vertexIndex: u32, 
    @builtin(instance_index) instanceIndex: u32
) -> VertexShaderOut {
    let instance: Instance = instances[instanceIndex];
    let vertex: Vertex = vertices[vertexIndex];
    let position: vec3f = vertex.position + instance.position;
    var out: VertexShaderOut;
    out.clipspace = uniforms.viewProjection * vec4f(position, 1);
    out.worldspace = position;
    return out;
}

@fragment fn fs(
    in: VertexShaderOut
) -> @location(0) vec4f {
    let index: u32 = u32(in.clipspace.y) * SCREEN_WIDTH + u32(in.clipspace.x);
    let normal: vec3f = normalize(cross(dpdx(in.worldspace), dpdy(in.worldspace)));
    let color: vec3f = normal * 0.5 + 0.5;
    let depth: f32 = in.clipspace.z;
    let value: u32 = pack_depth_color(depth, color);
    atomicMax(&frame[index], value);
    return vec4f(0, 0, 0, 0);
}

fn pack_depth_color(depth: f32, color: vec3f) -> u32 {
    let d = pack_depth(depth);
    let c = pack_rgb332(color);
    return (d << 8) | c;
}

fn pack_depth(depth: f32) -> u32 {
    let d: f32 = 1 - clamp(depth, 0, 1);
    return u32(d * 16777215);
}

fn pack_rgb332(color: vec3f) -> u32 {
    let r: u32 = u32(clamp(color.r, 0, 1) * 7);
    let g: u32 = u32(clamp(color.g, 0, 1) * 7);
    let b: u32 = u32(clamp(color.b, 0, 1) * 3);
    return (r << 5) | (g << 2) | b;
}