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
    position: vec3f
};

struct VertexShaderOut {
    @builtin(position) position: vec4f,
    @location(0) positionWorld: vec3f
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> vertecies: array<Vertex>;
@group(0) @binding(2) var<storage, read> instances: array<Instance>;

@vertex fn vs(@builtin(vertex_index) vertexIndex: u32, @builtin(instance_index) instanceIndex: u32) -> VertexShaderOut {
    let vertex: Vertex = vertecies[vertexIndex];
    let instance: Instance = instances[instanceIndex];

    var positionWorld: vec3f = vertex.position;
    positionWorld += instance.position;
    
    var out: VertexShaderOut;
    out.position = uniforms.viewProjection * vec4f(positionWorld, 1);
    out.positionWorld = positionWorld;
    return out;
}

@fragment fn fs(in: VertexShaderOut) -> @location(0) vec4f {
    var position: vec3f = in.positionWorld;
    let normal: vec3f = normalize(cross(dpdx(position), dpdy(position)));
    return vec4f(normal * 0.5 + 0.5, 1.0);
}