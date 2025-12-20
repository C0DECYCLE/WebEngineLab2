/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger
 */

struct Operation {
    initial: u32,
}

struct Queue {
    ringbuffer: array<u32, 65536>, //33554432
    tickets: array<atomic<u32>, 65536>, //33554432
    head: atomic<u32>, // should start 0
    tail: atomic<u32>, // should start 1
    count: atomic<i32>, // should start 1
};

const N = 65536u; // ringbuffer length, at most 2^16 //33554432u
const MaxThreads = 1048576u; // workgroup_size * dispatched amount
const failure = 4294967295u; // max u32

@group(0) @binding(0) var<storage, read_write> data: array<u32>;
@group(0) @binding(1) var<storage, read_write> operation: Operation;
@group(0) @binding(2) var<storage, read_write> queue: Queue;

fn ensureEnqueue() -> bool {
    var num = atomicLoad(&queue.count);
    var ensurance = false;
    loop {
        if (ensurance || num >= i32(N)) {
            break;
        }
        ensurance = atomicAdd(&queue.count, 1) < i32(N);
        if (!ensurance) {
            num = atomicSub(&queue.count, 1) - 1;
        }
    }
    return ensurance;
}

fn ensureDequeue() -> bool {
    var num = atomicLoad(&queue.count);
    var ensurance = false;
    loop {
        if (ensurance || num <= 0) {
            break;
        }
        ensurance = atomicSub(&queue.count, 1) > 0;
        if (!ensurance) {
            num = atomicAdd(&queue.count, 1) + 1;
        }
    }
    return ensurance;
}

fn waitForTicket(pos: u32, expected_ticket: u32) {
    loop {
        storageBarrier();
        if (atomicLoad(&queue.tickets[pos]) == expected_ticket) {
            break;
        }
    }
}

fn putData(element: u32) {
    var pos = atomicAdd(&queue.tail, 1u);
    var p = pos % N;
    waitForTicket(p, 2u*(pos/N));
    queue.ringbuffer[p] = element;
    storageBarrier();
    atomicStore(&queue.tickets[p], 2u * (pos/N) + 1u);
}

fn readData() -> u32 {
    var pos = atomicAdd(&queue.head, 1u);
    var p = pos % N;
    waitForTicket(p, 2u*(pos/N) + 1u);
    var element = queue.ringbuffer[p];
    storageBarrier();
    atomicStore(&queue.tickets[p], 2u*((pos+N)/N));
    return element;
}

fn enqueue(element: u32) -> bool {
    loop {
        if (ensureEnqueue()) {
            break;
        }
        var head = atomicLoad(&queue.head);
        var tail = atomicLoad(&queue.tail);
        if (N <= (tail-head) && (tail-head) < N + (MaxThreads/2u)) {
            return false;
        }
        storageBarrier();
    }
    putData(element);
    return true;
}

fn dequeue() -> u32 {
    loop {
        if (ensureDequeue()) {
            break;
        }
        var head = atomicLoad(&queue.head);
        var tail = atomicLoad(&queue.tail);
        if (N + (MaxThreads/2u) <= (tail-head) - 1u) {
            return failure;
        }
        storageBarrier();
    }
    return readData();
}

override WORKGROUP_SIZE_1D: u32;

@compute @workgroup_size(WORKGROUP_SIZE_1D) fn cs(
    @builtin(global_invocation_id) globalInvocationId: vec3u
) {
    let index: u32 = globalInvocationId.x;
    /*
    //loop {
        if (index == 0) {
            for (var i: u32 = 0; i < operation.initial; i++) {
                enqueue(i * 2);
            }
        }
        data[0] = data[0];
        //storageBarrier();
        //workgroupBarrier();
        //break;
    //}
    */
    if (index == 0) {
        enqueue(0);
        enqueue(2);
        enqueue(4);
        enqueue(6);
    }
}