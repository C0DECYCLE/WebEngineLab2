/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, January 2026
 */

#include structs.wgsl;
#include depthColorPack.wgsl;

override SCREEN_WIDTH: u32;
override SCREEN_HEIGHT: u32;

const WORKGROUP_SIZE: u32 = 64;

@group(0) @binding(0) var<storage, read> camera: Camera;
@group(0) @binding(1) var<storage, read> vertices: array<Vertex>;
@group(0) @binding(2) var<storage, read_write> args: DispatchWorkgroupsIndirect;
@group(0) @binding(3) var<storage, read_write> cache: array<SoftwareCache>;

@compute @workgroup_size(WORKGROUP_SIZE) fn cs(
    @builtin(global_invocation_id) globalInvocationId: vec3u
) {
    let index: u32 = globalInvocationId.x * 1;
    if (index >= arrayLength(&vertices)) {
        return;
    }
    let vertex: Vertex = vertices[index];
    let worldspace: vec3f = vertex.position + vec3f(4.5, 0, 4.5);
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
    let color: vec3f = vertex.color;
    let depth: f32 = ndc.z;
    let value: u32 = pack_depth_color(depth, color);
    let slot: u32 = atomicAdd(&args.workgroupCountX, 1);
    cache[slot] = SoftwareCache(screenspace, value);
}