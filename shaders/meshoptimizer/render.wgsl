/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, January 2025
 */

struct Uniforms {
    viewProjection: mat4x4f,
};

struct Task {
    instanceIndex: u32,
    meshletIndex: u32,
};

struct Instance {
    position: vec3f,
    meshletsOffset: u32,
    meshletsCount: u32,
};

struct Vertex {
    position: vec3f,
};

struct VertexShaderOut {
    @builtin(position) position: vec4f,
    @interpolate(flat) @location(0) color: vec4f,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> vertices: array<Vertex>;
@group(0) @binding(2) var<storage, read> instances: array<Instance>;
@group(0) @binding(3) var<storage, read> tasks: array<Task>;

@vertex fn vs(@builtin(vertex_index) vertexIndex: u32, @builtin(instance_index) instanceIndex: u32) -> VertexShaderOut {
    let task: Task = tasks[instanceIndex];
    let instance: Instance = instances[task.instanceIndex];
    let vertex: Vertex = vertices[task.meshletIndex * 128 * 3 + vertexIndex];

    var position: vec3f = vertex.position;
    position += instance.position;
    
    var out: VertexShaderOut;
    out.position = uniforms.viewProjection * vec4f(position, 1);
    out.color = vec4f(rndColor(f32(task.instanceIndex + task.meshletIndex + vertexIndex)), 1);
    return out;
}

@fragment fn fs(in: VertexShaderOut) -> @location(0) vec4f {
    return in.color;
}

fn rndColor(value: f32) -> vec3f {
    return fract(vec3f(value * 0.1443, value * 0.6841, value * 0.7323)) * 0.75 + 0.25;
}