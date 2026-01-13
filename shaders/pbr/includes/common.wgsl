/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger
 */

fn unpack(pack: VertexPack) -> Vertex {
  let position: vec3f = vec3f(pack.px, pack.py, pack.pz);
  let normal: vec3f = vec3f(pack.nx, pack.ny, pack.nz);
  let uv: vec2f = vec2f(pack.u, pack.v);
  return Vertex(position, normal, uv);
}

fn derive(position: vec3f, normal: vec3f, uv: vec2f) -> mat3x3f {
    let dposition1: vec3f = dpdx(position);
    let dposition2: vec3f = dpdy(position);
    let duv1: vec2f = dpdx(uv);
    let duv2: vec2f = dpdy(uv);
    let dposition2perp: vec3f = cross(dposition2, normal);
    let dposition1perp: vec3f = cross(normal, dposition1);
    let tangent: vec3f = dposition2perp * duv1.x + dposition1perp * duv2.x;
    let bitangent: vec3f = dposition2perp * duv1.y + dposition1perp * duv2.y;
    let invmax: f32 = inverseSqrt(max(dot(tangent, tangent), dot(bitangent, bitangent)));
    return mat3x3f(tangent * invmax, bitangent * invmax, normal);
}



fn srgbToLinearF32(srgb: f32) -> f32 {
  return pow(srgb, 2.2);
}

fn linearToSrgbF32(linear: f32) -> f32 {
  return pow(linear,  1 / 2.2);
}

fn srgbToLinearVec3f(srgb: vec3f) -> vec3f {
  return pow(srgb, vec3f(2.2));
}

fn linearToSrgbVec3f(linear: vec3f) -> vec3f {
  return pow(linear, vec3f(1 / 2.2));
}



fn tonemapReinhard(color: vec3f) -> vec3f {
    return color / (color + vec3f(1));
}

fn tonemapACES(color: vec3f) -> vec3f {
    let n: vec3f = color * (2.51 * color + vec3f(0.03));
    let z: vec3f = color * (2.43 * color + vec3f(0.59)) + vec3f( 0.14);
    return clamp(n / z, vec3f(0), vec3f(1));
}

fn tonemapAGX(color: vec3f) -> vec3f {
    var v: vec3f = color;
    let agx_inset_matrix: mat3x3f = mat3x3f(
        0.84247906, 0.04232824, 0.04237565,
        0.07843360, 0.87846864, 0.07843360,
        0.07922375, 0.07918443, 0.87914429
    );
    v = agx_inset_matrix * v;
    v = clamp(log2(max(v, vec3f(1e-10))), vec3f(-12.47393), vec3f(4.026069));
    v = (v + 12.47393) / (4.026069 + 12.47393);
    let x2: vec3f = v * v;
    let x4: vec3f = x2 * x2;
    v = (15.5 * x4 * x2) - (40.14 * x4 * v) + (31.96 * x4) - (6.868 * x2 * v) + (0.4298 * x2) + (0.1191 * v) - 0.00232;
    let agx_outset_matrix: mat3x3f = mat3x3f(
        1.19682103, -0.05289685, -0.05297163,
        -0.09802088, 1.15190312, -0.09804345,
        -0.09902974, -0.09899117, 1.15107367
    );
    v = agx_outset_matrix * v;
    return clamp(v, vec3f(0), vec3f(1));
}

fn tonemapPBRNeutral(color: vec3f) -> vec3f {
    let startCompression: f32 = 0.8;
    let desaturation: f32 = 0.15;
    let x: f32 = min(color.r, min(color.g, color.b));
    let peak: f32 = max(color.r, max(color.g, color.b));
    if (peak < startCompression) {
        return color;
    }
    let d: f32 = 1 - startCompression;
    let newPeak: f32 = 1 - d * d / (peak + d - startCompression);
    let scale: f32 = newPeak / peak;
    let compressed: vec3f = color * scale;
    let g: f32 = dot(compressed, vec3(0.299, 0.587, 0.114));
    return mix(compressed, vec3(g), desaturation);
}



fn fresnelSchlick(cosTheta: f32, F0: vec3<f32>) -> vec3<f32> {
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

fn distributionGGX(N: vec3<f32>, H: vec3<f32>, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let NdotH = max(dot(N, H), 0.0);
    let denom = (NdotH * NdotH * (a2 - 1.0) + 1.0);
    return a2 / (3.14159265 * denom * denom);
}

fn geometrySchlickGGX(NdotV: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}

fn geometrySmith(N: vec3<f32>, V: vec3<f32>, L: vec3<f32>, roughness: f32) -> f32 {
    let NdotV = max(dot(N, V), 0.0);
    let NdotL = max(dot(N, L), 0.0);
    let ggxV = geometrySchlickGGX(NdotV, roughness);
    let ggxL = geometrySchlickGGX(NdotL, roughness);
    return ggxV * ggxL;
}

fn SheenBRDF(N: vec3<f32>, V: vec3<f32>, L: vec3<f32>, sheenColor: vec3<f32>) -> vec3<f32> {
    let NdotL = max(dot(N, L), 0.0);
    let NdotV = max(dot(N, V), 0.0);
    return sheenColor * (NdotL / (NdotL + NdotV - NdotL * NdotV));
}