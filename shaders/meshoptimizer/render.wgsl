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
    @interpolate(flat) @location(0) color: vec4f,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> vertecies: array<Vertex>;
//@group(0) @binding(2) var<storage, read> instances: array<Instance>;

@vertex fn vs(@builtin(vertex_index) vertexIndex: u32, @builtin(instance_index) instanceIndex: u32) -> VertexShaderOut {
    let vertex: Vertex = vertecies[vertexIndex];
    //let instance: Instance = instances[instanceIndex];

    var position: vec3f = vertex.position;
    //position += instance.position;
    
    var out: VertexShaderOut;
    out.position = uniforms.viewProjection * vec4f(position, 1);
    out.color = vec4f(
        (f32(instanceIndex + 1) / 15) % 1,
        (f32(instanceIndex + 1) / 10) % 1,
        (f32(instanceIndex + 1) / 5) % 1,
        1
    );
    return out;
}

@fragment fn fs(in: VertexShaderOut) -> @location(0) vec4f {
    return vec4f(in.position.z, in.position.z, in.position.z, 2) / 2;
}