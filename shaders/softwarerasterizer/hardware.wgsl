/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger
 */

#include structs.wgsl;
#include depthColorPack.wgsl;

override SCREEN_WIDTH: u32;
override SCREEN_HEIGHT: u32;

@group(0) @binding(0) var<storage, read> camera: Camera;
@group(0) @binding(1) var<storage, read> vertices: array<Vertex>;
@group(0) @binding(2) var<storage, read> instances: array<Instance>;
@group(0) @binding(3) var<storage, read> instanceIndices: array<u32>;
@group(0) @binding(4) var<storage, read_write> frame: array<atomic<u32>>;

@vertex fn vs(
    @builtin(vertex_index) vertexIndex: u32, 
    @builtin(instance_index) instanceIndex: u32
) -> HardwareCache {
    let vertex: Vertex = vertices[vertexIndex];
    let instance: Instance = instances[instanceIndices[instanceIndex]];
    let worldspace: vec3f = vertex.position + instance.position;
    let position: vec4f = camera.viewProjection * vec4f(worldspace, 1);
    return HardwareCache(position, worldspace, vertex.color);
}

@fragment fn fs(
    cache: HardwareCache
) -> @location(0) vec4f {
    let position: vec4f = cache.position;
    let worldspace: vec3f = cache.worldspace;
    let normal: vec3f = normalize(cross(dpdx(worldspace), dpdy(worldspace)));
    let color: vec3f = cache.color;
    let depth: f32 = position.z;
    let value: u32 = pack_depth_color(depth, color);
    let pixel: u32 = u32(position.y) * SCREEN_WIDTH + u32(position.x);
    atomicMax(&frame[pixel], value);
    return vec4f(0, 0, 0, 0);
}