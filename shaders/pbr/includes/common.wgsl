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