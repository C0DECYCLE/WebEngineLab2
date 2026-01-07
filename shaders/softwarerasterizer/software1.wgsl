/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger
 */

#include structs.wgsl;
#include depthColorPack.wgsl;

override SCREEN_WIDTH: u32;
override SCREEN_HEIGHT: u32;

const VOXEL_SIZE: u32 = 8;

@group(0) @binding(0) var<storage, read> cache: array<SoftwareCache>;
@group(0) @binding(1) var<storage, read_write> frame: array<atomic<u32>>;

@compute @workgroup_size(VOXEL_SIZE, VOXEL_SIZE) fn cs(
    @builtin(workgroup_id) workgroupId: vec3u,
    @builtin(local_invocation_id) localInvocationId: vec3u
) {
    let groupIndex: u32 = workgroupId.x;
    let localIndex: vec2u = localInvocationId.xy;
    let cache: SoftwareCache = cache[groupIndex];
    let screenspace: vec2u = cache.screenspace;
    let value: u32 = cache.value;
    let x: i32 = i32(screenspace.x + localIndex.x - VOXEL_SIZE / 2);
    let y: i32 = i32(screenspace.y + localIndex.y - VOXEL_SIZE / 2);
    if (x < 0 || y < 0 || x >= i32(SCREEN_WIDTH) || y >= i32(SCREEN_HEIGHT)) {
        return;
    }
    let pixel: u32 = u32(y) * SCREEN_WIDTH + u32(x);
    atomicMax(&frame[pixel], value);
}