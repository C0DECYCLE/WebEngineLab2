/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, January 2026
 */

struct Instance {
    position: vec3f,
    meshletsOffset: u32,
    meshletsCount: u32,
};

struct Task {
    instanceIndex: u32,
    meshletIndex: u32,
};

struct Indirect {
    vertexCount: u32,
    instanceCount: atomic<u32>,
    firstVertex: u32,
    firstInstance: u32
};

@group(0) @binding(0) var<storage, read> instances: array<Instance>;
@group(0) @binding(1) var<storage, read_write> tasks: array<Task>;
@group(0) @binding(2) var<storage, read_write> indirect: Indirect;

@compute @workgroup_size(1) fn cs(@builtin(global_invocation_id) globalInvocationId: vec3<u32>) {
    let instanceIndex: u32 = globalInvocationId.x;
    let instance: Instance = instances[instanceIndex];
    for (var i: u32 = 0; i < instance.meshletsCount; i++) {
        let meshletIndex: u32 = instance.meshletsOffset + i;
        tasks[atomicAdd(&indirect.instanceCount, 1)] = Task(instanceIndex, meshletIndex);
    }
}