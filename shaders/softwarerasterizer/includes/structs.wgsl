/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger
 */

struct Camera {
    position: vec3f,
    viewProjection: mat4x4f,
}

struct Voxel {
    position: vec3f,
    color: vec3f,
}

struct Vertex {
    position: vec3f,
    color: vec3f,
}

struct Instance {
    position: vec3f,
}

struct DispatchWorkgroupsIndirect {
    workgroupCountX: atomic<u32>,
    workgroupCountY: atomic<u32>,
    workgroupCountZ: atomic<u32>,
}

struct DrawIndexedIndirect {
    indexCount: atomic<u32>,
    instanceCount: atomic<u32>,
    firstIndex: atomic<u32>,
    baseVertex: atomic<u32>,
    firstInstance: atomic<u32>,
}

struct SoftwareCache {
    screenspace: vec2u,
    value: u32,
}

struct HardwareCache {
    @builtin(position) position: vec4f,
    @location(0) worldspace: vec3f,
    @location(1) color: vec3f,
}