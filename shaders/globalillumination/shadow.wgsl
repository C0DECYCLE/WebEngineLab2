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

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> vertices: array<Vertex>;
@group(0) @binding(2) var<storage, read> instances: array<Instance>;

@vertex fn vs(@builtin(vertex_index) vertexIndex: u32, @builtin(instance_index) instanceIndex: u32) -> @builtin(position) vec4<f32> {
    let vertex: Vertex = vertices[vertexIndex];
    let instance: Instance = instances[instanceIndex];
    let position: vec3f = vertex.position * instance.scaling + instance.position;
    return uniforms.lightViewProjection * vec4f(position, 1);
}