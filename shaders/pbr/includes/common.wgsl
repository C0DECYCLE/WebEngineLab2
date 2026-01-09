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