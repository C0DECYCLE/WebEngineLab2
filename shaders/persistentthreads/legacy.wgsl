/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger
 */

struct Operation {
    offset: u32,
    length: u32,
}

override WORKGROUP_SIZE_1D: u32;

@group(0) @binding(0) var<storage, read_write> data: array<u32>;
@group(1) @binding(0) var<storage, read> operation: Operation;

@compute @workgroup_size(WORKGROUP_SIZE_1D) fn cs(
    @builtin(global_invocation_id) globalInvocationId: vec3u
) {
    let index: u32 = globalInvocationId.x;
    let dstLength: u32 = operation.length;
    if (index >= dstLength) {
        return;
    }
    let srcLength: u32 = dstLength * 2;
    let dstOffset: u32 = operation.offset;
    let srcOffset: u32 = dstOffset - srcLength;
    let a: u32 = data[srcOffset + index * 2 + 0];
    let b: u32 = data[srcOffset + index * 2 + 1];
    data[dstOffset + index] = a + b;
}