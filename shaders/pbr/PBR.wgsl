/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger
 */

#include consts.wgsl;
#include structs.wgsl;
#include common.wgsl;

@group(0) @binding(0) var<storage, read> camera: Camera;
@group(0) @binding(1) var<storage, read> vertices: array<VertexPack>;
@group(0) @binding(2) var textureSampler: sampler;
@group(0) @binding(3) var albedoTexture: texture_2d<f32>;
@group(0) @binding(4) var normalTexture: texture_2d<f32>;
@group(0) @binding(5) var roughnessTexture: texture_2d<f32>;
@group(0) @binding(6) var metalnessTexture: texture_2d<f32>;
@group(0) @binding(7) var ambientOcclusionTexture: texture_2d<f32>;
@group(0) @binding(8) var cavityTexture: texture_2d<f32>;
@group(0) @binding(9) var fuzzTexture: texture_2d<f32>;

@vertex fn vs(
    @builtin(vertex_index) vertexIndex: u32
) -> Rasterize {
    let vertex: Vertex = unpack(vertices[vertexIndex]);
    let position: vec3f = vertex.position;
    let normal: vec3f = normalize(vertex.normal);
    let uv: vec2f = vertex.uv;
    let point: vec4f = camera.viewProjection * vec4f(position, 1);
    return Rasterize(point, position, normal, uv);
}

@fragment fn fs(
    rasterize: Rasterize
) -> @location(0) vec4f {

    let albedoSample: vec3f = textureSample(albedoTexture, textureSampler, rasterize.uv).rgb;
    let normalSample: vec3f = textureSample(normalTexture, textureSampler, rasterize.uv).rgb;
    let roughnessSample: f32 = textureSample(roughnessTexture, textureSampler, rasterize.uv).r;
    let metalnessSample: f32 = textureSample(metalnessTexture, textureSampler, rasterize.uv).r;
    let ambientOcclusionSample: f32 = textureSample(ambientOcclusionTexture, textureSampler, rasterize.uv).r;
    let cavitySample: f32 = textureSample(cavityTexture, textureSampler, rasterize.uv).r;
    let fuzzSample: f32 = textureSample(fuzzTexture, textureSampler, rasterize.uv).r;
    
    return vec4(albedoSample, 1);
    //return vec4(rasterize.normal * 0.5 + 0.5, 1);
}