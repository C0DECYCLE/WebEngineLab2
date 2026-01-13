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
@group(0) @binding(5) var specularTexture: texture_2d<f32>;
@group(0) @binding(6) var glossTexture: texture_2d<f32>;
@group(0) @binding(7) var ambientOcclusionTexture: texture_2d<f32>;
@group(0) @binding(8) var cavityTexture: texture_2d<f32>;

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
    let baseColorSample: vec3f = srgbToLinearVec3f(textureSample(baseColorTexture, textureSampler, rasterize.uv).rgb);
    let normalSample: vec3f = textureSample(normalTexture, textureSampler, rasterize.uv).rgb;
    let specularColor: vec3f = srgbToLinearVec3f(vec3f(1, 1, 1));
    let specularSample: f32 = srgbToLinearF32(textureSample(specularTexture, textureSampler, rasterize.uv).r);
    let glossSample: f32 = srgbToLinearF32(textureSample(glossTexture, textureSampler, rasterize.uv).r);
    let ambientOcclusionSample: f32 = srgbToLinearF32(textureSample(ambientOcclusionTexture, textureSampler, rasterize.uv).r);
    let cavitySample: f32 = srgbToLinearF32(textureSample(cavityTexture, textureSampler, rasterize.uv).r);

    let tbn: mat3x3f = derive(rasterize.position, normalize(rasterize.normal), rasterize.uv);
    let normal: vec3f = normalize(tbn * (normalSample * 2 - 1)); //return vec4f(normal * 0.5 + 0.5, 1);
    let light: vec3f = normalize(vec3f(1, 1, 1));
    let view: vec3f = normalize(camera.position - rasterize.position);
    let halfway: vec3f = normalize(light + view);

    let ambientStrength: f32 = ambientOcclusionSample * cavitySample;
    let ambient: vec3f = ambientStrength * baseColorSample;

    let diffuseStrength: f32 = max(dot(normal, light), 0) * cavitySample;
    let diffuse: vec3f = diffuseStrength * baseColorSample;

    let specularStrength: f32 = pow(max(dot(normal, halfway), 0), pow(2, glossSample * 7)) * specularSample;
    let specular: vec3f = specularStrength * specularColor;

    let color: vec3f = ambient + diffuse + specular;
    return vec4f(linearToSrgbVec3f(color), 1);
}