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

struct Vertex {
    position: vec3f,
    color: vec3f,
};

struct Instance {
    position: vec3f,
    scaling: vec3f,
    color: vec3f
};

struct VertexOut {
    @builtin(position) position: vec4f,
    @interpolate(flat) @location(0) color: vec3f,
    @location(1) world: vec3f
};

struct FragmentOut {
  @location(0) color: vec4f,
  @location(1) normal: vec4f,
  @location(2) position: vec4f,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> vertices: array<Vertex>;
@group(0) @binding(2) var<storage, read> instances: array<Instance>;

@vertex fn vs(@builtin(vertex_index) vertexIndex: u32, @builtin(instance_index) instanceIndex: u32) -> VertexOut {
    let vertex: Vertex = vertices[vertexIndex];
    let instance: Instance = instances[instanceIndex];
    let position: vec3f = vertex.position * instance.scaling + instance.position;
    var out: VertexOut;
    out.position = uniforms.viewProjection * vec4f(position, 1);
    out.color = vertex.color * instance.color;
    out.world = position;
    return out;
}

@fragment fn fs(in: VertexOut) -> FragmentOut {
    var out: FragmentOut;
    out.color = vec4f(in.color, 1);
    out.normal = vec4f(normalize(cross(dpdx(in.world), dpdy(in.world))), 1);
    out.position = vec4f(in.world, 1);
    return out;
}