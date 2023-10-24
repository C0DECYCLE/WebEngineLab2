/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, October 2023
 */

struct Instance {
    position: vec3f,
};

@group(0) @binding(0) var<storage, read_write> instances: array<Instance>;
 
@compute @workgroup_size(1, 1, 1) fn computeInstance(
    @builtin(global_invocation_id) id: vec3<u32>
) {
    let instanceIndex: u32 = id.x;
    var position: vec3f = vec3f(
        noise(f32(instanceIndex) * 0.456 + 213.534) * 10.0 - 5.0,
        noise(f32(instanceIndex) * 0.182 + 562.913) * 0.5,
        noise(f32(instanceIndex) * 0.915 - 610.812) * 10.0 - 5.0
    );
    instances[instanceIndex] = Instance(position);
}

fn rand(n: f32) -> f32 {
    return fract(sin(n) * 43758.5453123);
}

fn noise(p: f32) -> f32 {
    let fl = floor(p);
    return mix(rand(fl), rand(fl + 1.), fract(p));
}
