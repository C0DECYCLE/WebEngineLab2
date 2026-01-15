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
@group(0) @binding(3) var baseColorTexture: texture_2d<f32>;
@group(0) @binding(4) var normalTexture: texture_2d<f32>;
@group(0) @binding(5) var roughnessTexture: texture_2d<f32>;
@group(0) @binding(6) var metalnessTexture: texture_2d<f32>;

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
    let albedo: vec3f = textureSample(baseColorTexture, textureSampler, rasterize.uv).rgb;
    let normalMap: vec3f = textureSample(normalTexture, textureSampler, rasterize.uv).rgb;
    let roughness: f32 = textureSample(roughnessTexture, textureSampler, rasterize.uv).r;
    let metalness: f32 = textureSample(metalnessTexture, textureSampler, rasterize.uv).r;

    let TBN: mat3x3f = deriveTBN(rasterize.position, normalize(rasterize.normal), rasterize.uv);
    let N: vec3f = normalize(TBN * normalize(normalMap * 2 - 1));
    let V: vec3f = normalize(camera.position - rasterize.position);
    let L: vec3f = normalize(-vec3f(-1, -1, -1));
    let H: vec3f = normalize(V + L);
    let radiance: vec3f = vec3f(5);
    let HdotV: f32 = saturate(dot(H, V));
    let NdotL: f32 = saturate(dot(N, L)/* * 0.5 + 0.5*/);
    let NdotV: f32 = saturate(dot(N, V));

    let F0: vec3f = mix(vec3f(0.04), albedo, metalness);
    let NDF: f32 = distributionGGX(N, H, roughness);
    let G: f32 = geometrySmith(N, V, L, roughness);
    let F: vec3f = fresnelSchlick(HdotV, F0);

    let numerator: vec3f = NDF * G * F;
    let denominator: f32 = 4 * NdotV * NdotL + 0.0001;
    let specular: vec3f = numerator / denominator;

    let kS: vec3f = F;
    let kD: vec3f = (vec3f(1) - kS) * (1 - metalness);
    let color: vec3f = (kD * albedo / PI + specular) * radiance * NdotL;

    return vec4f(tonemapReinhard(color), 1);
}