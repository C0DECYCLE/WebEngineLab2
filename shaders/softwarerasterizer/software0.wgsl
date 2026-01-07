/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger
 */

#include structs.wgsl;
#include depthColorPack.wgsl;

override SCREEN_WIDTH: u32;
override SCREEN_HEIGHT: u32;

override WORKGROUP_SIZE_X: u32;

const WORKGROUP_SIZE_Y: u32 = 1;

@group(0) @binding(0) var<storage, read> camera: Camera;
@group(0) @binding(1) var<storage, read> voxels: array<Voxel>;
@group(0) @binding(2) var<storage, read> instances: array<Instance>;
@group(0) @binding(3) var<storage, read> instanceIndices: array<u32>;
@group(0) @binding(4) var<storage, read_write> args: DispatchWorkgroupsIndirect;
@group(0) @binding(5) var<storage, read_write> cache: array<SoftwareCache>;

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y) fn cs(
    @builtin(global_invocation_id) globalInvocationId: vec3u
) {
    let voxelIndex: u32 = globalInvocationId.x;
    let instanceIndex: u32 = globalInvocationId.y;
    if (voxelIndex >= arrayLength(&voxels)) {
        return;
    }
    let voxel: Voxel = voxels[voxelIndex];
    let instance: Instance = instances[instanceIndices[instanceIndex]];
    let worldspace: vec3f = voxel.position + instance.position;
    let clipspace: vec4f = camera.viewProjection * vec4f(worldspace, 1);
    if (clipspace.w <= 0) {
        return;
    }
    let ndc: vec3f = clipspace.xyz / clipspace.w;
    if (ndc.x < -1 || ndc.x > 1 || ndc.y < -1 || ndc.y > 1 || ndc.z <  0 || ndc.z > 1) {
        return;
    }
    let screenspace: vec2u = vec2u(
        u32((ndc.x * 0.5 + 0.5) * f32(SCREEN_WIDTH)),
        u32((1 - (ndc.y * 0.5 + 0.5)) * f32(SCREEN_HEIGHT))
    );
    let color: vec3f = voxel.color;
    let depth: f32 = ndc.z;
    let value: u32 = pack_depth_color(depth, color);
    let slot: u32 = atomicAdd(&args.workgroupCountX, 1);
    cache[slot] = SoftwareCache(screenspace, value);
}