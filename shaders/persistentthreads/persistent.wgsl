/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger
 */

const QUEUE_SIZE: u32 = 1024;
const QUEUE_MASK: u32 = QUEUE_SIZE - 1;

struct QueueItem {
    src: u32,
    dst: u32,
}

struct Queue {
    head: atomic<u32>,
    tail: atomic<u32>,
    activeCount: atomic<u32>,
    data: array<QueueItem, QUEUE_SIZE>,
}

fn enqueue(value: QueueItem) -> bool {
    let h: u32 = atomicAdd(&queue.head, 1);
    let t: u32 = atomicLoad(&queue.tail);
    if (h - t >= QUEUE_SIZE) {
        atomicSub(&queue.head, 1);
        return false;
    }
    queue.data[h & QUEUE_MASK] = value;
    atomicAdd(&queue.activeCount, 1);
    return true;
}

fn dequeue(value: ptr<function, QueueItem>) -> bool {
    let t: u32 = atomicAdd(&queue.tail, 1);
    let h: u32 = atomicLoad(&queue.head);
    if (t >= h) {
        atomicSub(&queue.tail, 1);
        return false;
    }
    *value = queue.data[t & QUEUE_MASK];
    return true;
}

override WORKGROUP_SIZE_1D: u32;
override INPUT_SIZE: u32;

@group(0) @binding(0) var<storage, read_write> data: array<u32>;
@group(0) @binding(1) var<storage, read_write> queue: Queue;

@compute @workgroup_size(WORKGROUP_SIZE_1D) fn cs(
    @builtin(global_invocation_id) globalInvocationId: vec3u
) {
    let index: u32 = globalInvocationId.x;

    if (index == 0) {
        for (var i: u32 = 0; i < INPUT_SIZE / 2; i++) {
            let newSrc: u32 = i * 2;
            let newDst: u32 = pow(2, ceil(log2(INPUT_SIZE * 2 - newSrc)) - 1);
            enqueue(QueueItem(newSrc, newDst));
        }
    }
    storageBarrier();
    workgroupBarrier();
    /*
    var current: QueueItem;
    for (var l: u32 = 0; l < 1024; l++) {

        if (dequeue(&current)) {
            
            let src: u32 = current.src; 
            let dst: u32 = current.dst;
            let a: u32 = data[src + 0];
            let b: u32 = data[src + 1];
            data[dst] = a + b;
            if (dst % 2 == 0 && data[dst + 1] != 0) {
                let newSrc: u32 = dst;
                let newDst: u32 = ?;
                enqueue(QueueItem(newSrc, newDst));
            }
            if (dst % 2 == 1 && data[dst - 1] != 0) {
                let newSrc: u32 = dst - 1;
                let newDst: u32 = ?;
                enqueue(QueueItem(newSrc, newDst));
            }
            
            atomicSub(&queue.activeCount, 1u);

        } else {
            if (atomicLoad(&queue.activeCount) == 0) {
                break;
            }
            // Optional: small yield/spin to reduce contention
        }
    }
    */


    data[0] = data[0];
    queue.data[0] = queue.data[0];
}