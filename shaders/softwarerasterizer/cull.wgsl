/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger
 */

#include structs.wgsl;
#include depthColorPack.wgsl;

override WORKGROUP_SIZE: u32;

const VOXEL_DISTANCE: f32 = 0;

@group(0) @binding(0) var<storage, read> camera: Camera;
@group(0) @binding(1) var<storage, read> instances: array<Instance>;
@group(0) @binding(2) var<storage, read_write> softwareArgs: DispatchWorkgroupsIndirect;
@group(0) @binding(3) var<storage, read_write> softwareInstances: array<u32>;
@group(0) @binding(4) var<storage, read_write> hardwareArgs: DrawIndexedIndirect;
@group(0) @binding(5) var<storage, read_write> hardwareInstances: array<u32>;

@compute @workgroup_size(WORKGROUP_SIZE) fn cs(
    @builtin(global_invocation_id) globalInvocationId: vec3u
) {
    let index: u32 = globalInvocationId.x;
    if (index >= arrayLength(&instances)) {
        return;
    }
    let instance: Instance = instances[index];
    let distance: f32 = length(instance.position - camera.position);
    if (distance >= VOXEL_DISTANCE) {
        let slot: u32 = atomicAdd(&softwareArgs.workgroupCountY, 1);
        softwareInstances[slot] = index;
        return;
    }
    let slot: u32 = atomicAdd(&hardwareArgs.instanceCount, 1);
    hardwareInstances[slot] = index;
}