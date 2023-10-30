/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, October 2023
 */

struct Instance {
    matrix: mat4x4f,
};

const spread: f32 = 30.0;

@group(0) @binding(0) var<storage, read_write> instances: array<Instance>;

@compute @workgroup_size(100, 1, 1) fn computeInstance(
    @builtin(global_invocation_id) id: vec3<u32>
) {
    let instanceIndex: u32 = id.x;
    let col: f32 = floor(f32(instanceIndex) / 100);
    let row: f32 = f32(instanceIndex % 100);
    var position: vec3f = vec3f(
        //row * 0.02 * spread - spread * 0.5,
        noise(f32(instanceIndex) * 0.456 + 213.534) * spread - spread * 0.5,
        0.0,
        //col * 0.02 * spread - spread * 0.5
        noise(f32(instanceIndex) * 0.915 - 610.812) * spread - spread * 0.5
    );
    //position.y = abs(perlinNoise2(position.xz * 0.05)) * 5.0;
    let radian: f32 = noise(f32(instanceIndex) * 0.712 + 918.782) * 6.3;
    let sc: f32 = abs(perlinNoise2(position.xz * 0.35)) * 2.0 + 1.0;
    let c: f32 = cos(radian);
    let s: f32 = sin(radian);
    let matrix: mat4x4f = mat4x4f(
         c * sc,      0, s * sc, position.x,
              0, 1 * sc,      0, position.y,
        -s * sc,      0, c * sc, position.z,
              0,      0,      0,          1
    );
    instances[instanceIndex] = Instance(matrix);
}

fn rand(n: f32) -> f32 {
    return fract(sin(n) * 43758.5453123);
}

fn noise(p: f32) -> f32 {
    let fl = floor(p);
    return mix(rand(fl), rand(fl + 1.), fract(p));
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