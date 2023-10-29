/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, October 2023
 */

struct Uniforms {
    viewProjection: mat4x4f,
    mode: u32, // 0: triangle, 1: normal, 2: grass
    time: f32
};

struct Vertex {
    position: vec3f,
};

struct Instance {
    matrix: mat4x4f,
};

struct VertexShaderOut {
    @builtin(position) position: vec4f,
    @location(0) worldPosition: vec3f,
    @interpolate(flat) @location(1) flatWorldPosition: vec3f,
    @location(2) selfHeight: f32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> vertecies: array<Vertex>;
@group(0) @binding(2) var<storage, read> instances: array<Instance>;

@vertex fn vs(
    @builtin(vertex_index) vertexIndex: u32,
    @builtin(instance_index) instanceIndex: u32
) -> VertexShaderOut {
    let vertex: Vertex = vertecies[vertexIndex];
    let instance: Instance = instances[instanceIndex];

    var position: vec3f = vertex.position;
    position.z += cos(position.y) - 1.0;
    position = (vec4f(position, 1.0) * instance.matrix).xyz;
    
    var windStrength: f32 = position.y * perlinNoise2(position.xz * 0.05 + uniforms.time * 0.001) * 0.35;
    var windDirection: vec2f = vec2f(cos(uniforms.time * 0.0001), sin(uniforms.time * 0.0001));
    position.x += windDirection.x * windStrength;
    position.z += windDirection.y * windStrength;

    var out: VertexShaderOut;
    out.position = uniforms.viewProjection * vec4f(position, 1.0);
    out.worldPosition = position;
    out.flatWorldPosition = position;
    out.selfHeight = vertex.position.y;
    return out;
}

@fragment fn fs(in: VertexShaderOut) -> @location(0) vec4f {
    if (uniforms.mode == 0) {
        return triangleShade(in.flatWorldPosition);
    }
    if (uniforms.mode == 1) {
        return normalShade(in.worldPosition);
    }
    if (uniforms.mode == 2) {
        return grassShade(in.worldPosition, in.selfHeight);
    }
    return vec4f(0.0, 0.0, 0.0, 1.0);
}

fn triangleShade(pos: vec3f) -> vec4f {
    let s: f32 = 100.0;
    let ox: f32 = pos.x * s;
    let oy: f32 = pos.y * s;
    let oz: f32 = pos.z * s;
    let r: f32 = noise3(vec3f(oy, oz, ox));
    let g: f32 = noise3(vec3f(oz, ox, oy));
    let b: f32 = noise3(vec3f(ox, oy, oz));
    return vec4f(r, g, b, 1.0);
}

fn normalShade(pos: vec3f) -> vec4f {
    let normal: vec3f = normalize(cross(dpdx(pos), dpdy(pos)));
    return vec4f(normal * 0.5 + 0.5, 1.0);
}

fn grassShade(pos: vec3f, hgt: f32) -> vec4f {
    /*
    let faceNormal: vec3f = normalize(cross(dpdx(pos), dpdy(pos)));
    let lightDirection: vec3f = normalize(vec3f(-0.5, -1.0, -0.5));

    let theta: f32 = dot(faceNormal, lightDirection);
    let shade: f32 = max(0.0, theta * 0.5 + 0.5);
    */

    let bottomColor: vec3f = vec3f(0.0, 0.2, 0.0);
    let topColor: vec3f = vec3f(0.3, 1.0, 0.5);
    let highColor: vec3f = vec3f(0.8, 1.0, 0.0);
    let finalColor: vec3f = mix(bottomColor, mix(topColor, highColor, hgt), hgt);
    return vec4f(finalColor, 1.0);
}

fn rand(n: f32) -> f32 {
    return fract(sin(n) * 43758.5453123);
}

fn noise(p: f32) -> f32 {
    let fl = floor(p);
    return mix(rand(fl), rand(fl + 1.), fract(p));
}

fn mod289(x: vec4f) -> vec4f {
    return x - floor(x * (1. / 289.)) * 289.;
}

fn perm4(x: vec4f) -> vec4f {
    return mod289(((x * 34.) + 1.) * x);
}

fn noise3(p: vec3f) -> f32 {
    let a: vec3f = floor(p);
    var d: vec3f = p - a;
    d = d * d * (3. - 2. * d);
    let b: vec4f = a.xxyy + vec4f(0., 1., 0., 1.);
    let k1: vec4f = perm4(b.xyxy);
    let k2: vec4f = perm4(k1.xyxy + b.zzww);
    let c: vec4f = k2 + a.zzzz;
    let k3: vec4f = perm4(c);
    let k4: vec4f = perm4(c + 1.);
    let o1: vec4f = fract(k3 * (1. / 41.));
    let o2: vec4f = fract(k4 * (1. / 41.));
    let o3: vec4f = o2 * d.z + o1 * (1. - d.z);
    let o4: vec2f = o3.yw * d.x + o3.xz * (1. - d.x);
    return o4.y * d.y + o4.x * (1. - d.y);
}

fn permute4(x: vec4f) -> vec4f { return ((x * 34. + 1.) * x) % vec4f(289.); }
fn fade2(t: vec2f) -> vec2f { return t * t * t * (t * (t * 6. - 15.) + 10.); }

fn perlinNoise2(P: vec2f) -> f32 {
    var Pi: vec4f = floor(P.xyxy) + vec4f(0., 0., 1., 1.);
    let Pf = fract(P.xyxy) - vec4f(0., 0., 1., 1.);
    Pi = Pi % vec4f(289.); // To avoid truncation effects in permutation
    let ix = Pi.xzxz;
    let iy = Pi.yyww;
    let fx = Pf.xzxz;
    let fy = Pf.yyww;
    let i = permute4(permute4(ix) + iy);
    var gx: vec4f = 2. * fract(i * 0.0243902439) - 1.; // 1/41 = 0.024...
    let gy = abs(gx) - 0.5;
    let tx = floor(gx + 0.5);
    gx = gx - tx;
    var g00: vec2f = vec2f(gx.x, gy.x);
    var g10: vec2f = vec2f(gx.y, gy.y);
    var g01: vec2f = vec2f(gx.z, gy.z);
    var g11: vec2f = vec2f(gx.w, gy.w);
    let norm = 1.79284291400159 - 0.85373472095314 *
        vec4f(dot(g00, g00), dot(g01, g01), dot(g10, g10), dot(g11, g11));
    g00 = g00 * norm.x;
    g01 = g01 * norm.y;
    g10 = g10 * norm.z;
    g11 = g11 * norm.w;
    let n00 = dot(g00, vec2f(fx.x, fy.x));
    let n10 = dot(g10, vec2f(fx.y, fy.y));
    let n01 = dot(g01, vec2f(fx.z, fy.z));
    let n11 = dot(g11, vec2f(fx.w, fy.w));
    let fade_xy = fade2(Pf.xy);
    let n_x = mix(vec2f(n00, n01), vec2f(n10, n11), vec2f(fade_xy.x));
    let n_xy = mix(n_x.x, n_x.y, fade_xy.y);
    return 2.3 * n_xy;
}