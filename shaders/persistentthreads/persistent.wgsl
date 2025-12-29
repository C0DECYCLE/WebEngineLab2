/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger
 */
 
/* WORKGROUP */

override WORKGROUP_THREADS: u32;

var<workgroup> any_all: atomic<u32>;

fn workgroupAny(LID: u32, condition: bool) -> bool {
    workgroupBarrier();
    if (LID == 0) {
        atomicStore(&any_all, 0);
    }
    workgroupBarrier();
    if (condition) {
        atomicStore(&any_all, 1);
    }
    workgroupBarrier();
    return workgroupUniformLoad(&any_all) == 1;
}

fn workgroupAll(LID: u32, condition: bool) -> bool {
    workgroupBarrier();
    if (LID == 0) {
        atomicStore(&any_all, 0);
    }
    workgroupBarrier();
    if (condition) {
        atomicAdd(&any_all, 1);
    }
    workgroupBarrier();
    return workgroupUniformLoad(&any_all) == WORKGROUP_THREADS;
}

/* STACK */

const STACK_CAPACITY: i32 = 2048; //more? limit 4096 - 1

const LOOP_LIMIT: u32 = 1024 * 1024 * 1024; //1024; //4294967295;

struct Stack {
    top: atomic<i32>,
    buffer: array<u32, STACK_CAPACITY>,
}

var<workgroup> stack: Stack;

fn push(value: u32) -> bool {
    let old: i32 = atomicAdd(&stack.top, 1);
    if (old >= STACK_CAPACITY) { 
        atomicSub(&stack.top, 1);
        return false; 
    }
    stack.buffer[old] = value;
    return true;
}

fn pop(value: ptr<function, u32>) -> bool {
    let old: i32 = atomicSub(&stack.top, 1);
    if (old <= 0) { 
        atomicAdd(&stack.top, 1);
        return false; 
    }
    *value = stack.buffer[old - 1];
    return true;
}

/* DISPATCH */

override INPUT_SIZE: u32;

@group(0) @binding(0) var<storage, read_write> data: array<u32>;
@group(0) @binding(1) var<storage, read_write> debug: Stack;

@compute @workgroup_size(WORKGROUP_THREADS) fn cs(
    @builtin(local_invocation_index) LID: u32
) {
    /* INITIALIZE */

    if (LID == 0) {
        atomicStore(&stack.top, i32(INPUT_SIZE / 2));
        for (var i: u32 = 0; i < INPUT_SIZE / 2; i++) {
            let initial: u32 = INPUT_SIZE + i;
            stack.buffer[i] = initial;
        }
    }

    workgroupBarrier();

    /* PERSISTENT */
    
    for (var l0: u32 = 0; l0 < LOOP_LIMIT; l0++) {
        
        var current: u32;
        let success: bool = pop(&current);
        
        if (workgroupAll(LID, !success)) {
            break;
        }

        if (success) {
            let a: u32 = data[(current * 2) - (INPUT_SIZE * 2) + 0];
            let b: u32 = data[(current * 2) - (INPUT_SIZE * 2) + 1];
            data[current] = a + b;
        }    

        workgroupBarrier();

        if (success && current % 2 == 0 && data[current + 1] != 0) {
            let next: u32 = (current + (INPUT_SIZE * 2)) / 2;
            push(next);
        }

        workgroupBarrier();
    }

    workgroupBarrier();

    /* DEBUG */

    if (LID == 0) {
        atomicStore(&debug.top, atomicLoad(&stack.top));
        debug.buffer = stack.buffer;
    }
}