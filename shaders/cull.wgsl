/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, October 2023
 */

struct Uniforms {
    viewProjection: mat4x4f,
    time: f32
    //mode: u32, // 0: triangle, 1: normal, 2: grass
};

struct Instance {
    matrix: mat4x4f,
};

struct Indirect {
    vertexCount: u32,
    instanceCount: atomic<u32>,
    firstVertex: u32,
    firstInstance: u32
};

@group(0) @binding(0) var<storage, read> instances: array<Instance>;
@group(0) @binding(1) var<storage, read_write> draw: Indirect;
@group(0) @binding(2) var<storage, read_write> culled: array<Instance>;
@group(0) @binding(3) var<uniform> uniforms: Uniforms;

@compute @workgroup_size(100, 1, 1) fn cullInstance(
    @builtin(global_invocation_id) id: vec3<u32>
) {
    let instanceIndex: u32 = id.x;
    let instance: Instance = instances[instanceIndex];

    var position: vec3f = vec3f(instance.matrix[0][3], instance.matrix[1][3] + 0.5, instance.matrix[2][3]);
    var viewspace: vec4f = uniforms.viewProjection * vec4f(position, 1.0);
    
    var clipspace: vec3f = viewspace.xyz;
    clipspace /= -viewspace.w;

    clipspace.x = clipspace.x / 2.0 + 0.5;
    clipspace.y = clipspace.y / 2.0 + 0.5;
    clipspace.z = -viewspace.w;

    if (clipspace.x > 0.05 && clipspace.x < 0.95 && clipspace.y > 0.05 && clipspace.y < 0.95) {
    //if (clipspace.x > 0.0 && clipspace.x < 1.0 && clipspace.y > -0.2 && clipspace.y < 1.2) {
        let index: u32 = atomicAdd(&draw.instanceCount, 1);
        culled[index] = instance;
    }
}