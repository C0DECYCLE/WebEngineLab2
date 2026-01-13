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
@group(0) @binding(6) var ambientOcclusionTexture: texture_2d<f32>;
@group(0) @binding(7) var cavityTexture: texture_2d<f32>;
@group(0) @binding(8) var fuzzTexture: texture_2d<f32>;

@vertex fn vs(
    @builtin(vertex_index) vertexIndex: u32
) -> Rasterize {
    let vertex: Vertex = unpack(vertices[vertexIndex]);
    let position: vec3f = vertex.position;
    let normal: vec3f = vertex.normal;
    let uv: vec2f = vertex.uv;
    let point: vec4f = camera.viewProjection * vec4f(position, 1);
    return Rasterize(point, position, normal, uv);
}

@fragment fn fs(
    rasterize: Rasterize
) -> @location(0) vec4f {
    const metallic: f32 = 0;
    const fuzzStrength: f32 = 1;
    const lightColor: vec3f = vec3f(1, 1, 1);
    const lightDirection: vec3f = vec3f(-1, -1, -1);

    // === Sample textures ===
    let albedoSample: vec3f = textureSample(albedoTexture, textureSampler, rasterize.uv).rgb;
    let normalSample: vec3f = textureSample(normalTexture, textureSampler, rasterize.uv).rgb;
    let roughnessSample: f32 = textureSample(roughnessTexture, textureSampler, rasterize.uv).r;
    let ambientOcclusionSample: f32 = textureSample(ambientOcclusionTexture, textureSampler, rasterize.uv).r;
    let cavitySample: f32 = textureSample(cavityTexture, textureSampler, rasterize.uv).r;
    let fuzzSample: f32 = textureSample(fuzzTexture, textureSampler, rasterize.uv).r;

    // === Normal mapping ===
    let TBN: mat3x3f = derive(rasterize.position, normalize(rasterize.normal), rasterize.uv);
    let N: vec3f = normalize(TBN * (normalSample * 2 - 1));

    let V: vec3f = normalize(camera.position - rasterize.position);
    let L: vec3f = normalize(-lightDirection);
    let H: vec3f = normalize(V + L);

    // === Base reflectance ===
    let F0: vec3f = mix(vec3f(0.04), albedoSample, metallic);

    // === Cook-Torrance BRDF ===
    let NDF: f32 = distributionGGX(N, H, roughnessSample);
    let G: f32 = geometrySmith(N, V, L, roughnessSample);
    let F: vec3f = fresnelSchlick(max(dot(H, V), 0), F0);

    let numerator: vec3f = NDF * G * F;
    let denom: f32 = 4 * max(dot(N, V), 0.001) * max(dot(N, L), 0.001);
    let specular: vec3f = numerator / denom;

    let kS: vec3f = F;
    let kD: vec3f = (vec3f(1) - kS) * (1 - metallic);

    let NdotL: f32 = max(dot(N, L), 0);

    // === Lighting ===
    var color: vec3f  = (kD * albedoSample / PI + specular) * lightColor * NdotL;

    // === Fuzz (cloth effect) ===
    let sheen = SheenBRDF(N, V, L, vec3f(fuzzSample)) * albedoSample * (1 - NdotL);
    color += sheen;//fuzzSample * fuzzStrength * albedoSample * (1 - NdotL);

    // === AO + cavity ===
    color *= ambientOcclusionSample * cavitySample;

    // === Tonemap + gamma ===
    color = tonemapPBRNeutral(color * 2);
    color = linearToSrgbVec3f(color);

    return vec4(color, 1.0);
}