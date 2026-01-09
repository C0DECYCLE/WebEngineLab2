/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger
 */

struct Camera {
    position: vec3f,
    viewProjection: mat4x4f,
}

struct VertexPack {
  px: f32, 
  py: f32,
  pz: f32,
  nx: f32,
  ny: f32,
  nz: f32,
  u: f32,
  v: f32,
}

struct Vertex {
  position: vec3f,
  normal: vec3f,
  uv: vec2f,
}

struct Rasterize {
    @builtin(position) point: vec4f,
    @location(0) position: vec3f,
    @location(1) normal: vec3f,
    @location(2) uv: vec2f,
}