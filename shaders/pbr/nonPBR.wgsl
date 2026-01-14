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
@group(0) @binding(6) var roughnessTexture: texture_2d<f32>;
@group(0) @binding(7) var ambientOcclusionTexture: texture_2d<f32>;
@group(0) @binding(8) var cavityTexture: texture_2d<f32>;

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
    let specularStrength: f32 = textureSample(specularTexture, textureSampler, rasterize.uv).r;
    let roughness: f32 = textureSample(roughnessTexture, textureSampler, rasterize.uv).r;
    let ao: f32 = textureSample(ambientOcclusionTexture, textureSampler, rasterize.uv).r;
    let cavity: f32 = textureSample(cavityTexture, textureSampler, rasterize.uv).r;

    let lightDir: vec3f = normalize(-vec3f(-1, -1, -1));
    let lightColor: vec3f = vec3f(1/*, 0.8, 0.6*/);
    let tbn: mat3x3f = deriveTBN(rasterize.position, normalize(rasterize.normal), rasterize.uv);
    let normal: vec3f = normalize(tbn * normalize(normalMap * 2 - 1));
    let viewDir: vec3f = normalize(camera.position - rasterize.position);
    let halfDir: vec3f = normalize(lightDir + viewDir);

    let ambient: vec3f = albedo * 0.25 * ao * cavity;

    let halfLambert: f32 = dot(normal, lightDir) * 0.5 + 0.5;
    let diffuse: vec3f = albedo * halfLambert;

    let gloss: f32 = 1 - roughness;
    let shininess: f32 = mix(8, 256, gloss);
    let reflectivity: f32 = max(dot(normal, halfDir), 0);
    let specular: f32 = pow(reflectivity, shininess) * specularStrength;

    return vec4f(ambient + (diffuse + specular) * lightColor, 1);
}