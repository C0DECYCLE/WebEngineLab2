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
    return atomicLoad(&any_all) == 1; //workgroupUniformLoad and no barrier
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
    return atomicLoad(&any_all) == WORKGROUP_THREADS; //workgroupUniformLoad and no barrier
}

/* QUEUE */

const QUEUE_CAPACITY: u32 = 1024; //more? limit 4096 - 2
const QUEUE_FAILURE: u32 = 4294967295;

const LOOP_LIMIT: u32 = 4294967295; //1024;

struct Queue {
    head: atomic<u32>,
    tail: atomic<u32>,
    ringbuffer: array<u32, QUEUE_CAPACITY>,
}

var<workgroup> queue: Queue;

fn enqueue(value: u32) -> bool {
    let head: u32 = atomicLoad(&queue.head);
    let tail: u32 = atomicLoad(&queue.tail);
    if (head - tail >= QUEUE_CAPACITY) { 
        return false; 
    }
    if (!atomicCompareExchangeWeak(&queue.head, head, head + 1).exchanged) {
        return false;
    }
    queue.ringbuffer[head % QUEUE_CAPACITY] = value;
    return true;
}

fn dequeue(value: ptr<function, u32>) -> bool {
    let head: u32 = atomicLoad(&queue.head);
    let tail: u32 = atomicLoad(&queue.tail);
    if (tail >= head) {
        return false;
    }
    if (!atomicCompareExchangeWeak(&queue.tail, tail, tail + 1).exchanged) {
        return false;
    }
    *value = queue.ringbuffer[tail % QUEUE_CAPACITY];
    return true;
}

/*
struct Queue {
    head: atomic<u32>,
    tail: atomic<u32>,
    count: atomic<i32>,
    ringbuffer: array<u32/*atomic<u32>*/, QUEUE_CAPACITY>,
    tickets: array<atomic<u32>, QUEUE_CAPACITY>,
    //
    //workLeft: atomic<u32>,
}

var<workgroup> queue: Queue;

fn ensureEnqueue() -> bool {
    var num: i32 = atomicLoad(&queue.count);
    var ensurance: bool = false;

    //loop {
    for (var l: u32 = 0; l < LOOP_LIMIT; l++) {
        if (ensurance || num >= i32(QUEUE_CAPACITY)) {
            break;
        }
        ensurance = atomicAdd(&queue.count, 1) < i32(QUEUE_CAPACITY);
        if (!ensurance) {
            num = atomicSub(&queue.count, 1) - 1;
        }
    }
    return ensurance;
}

fn ensureDequeue() -> bool {
    var num: i32 = atomicLoad(&queue.count);
    var ensurance: bool = false;

    //loop {
    for (var l: u32 = 0; l < LOOP_LIMIT; l++) {
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
    //loop {
    for (var l: u32 = 0; l < LOOP_LIMIT; l++) {
        //storageBarrier();
        workgroupBarrier();

        if (atomicLoad(&queue.tickets[pos]) == expected_ticket) {
            break;
        }
    }
}

fn putData(element: u32) {
    let pos: u32 = atomicAdd(&queue.tail, 1);
    let p: u32 = pos % QUEUE_CAPACITY;
    let expected: u32 = 2 * (pos / QUEUE_CAPACITY);
    waitForTicket(p, expected);
    queue.ringbuffer[p] = element;
    //storageBarrier();
    workgroupBarrier();
    atomicStore(&queue.tickets[p], expected + 1);
}

fn readData() -> u32 {
    let pos: u32 = atomicAdd(&queue.head, 1);
    let p: u32 = pos % QUEUE_CAPACITY;
    let expected: u32 = 2 * (pos / QUEUE_CAPACITY);
    waitForTicket(p, expected + 1);
    var element: u32 = queue.ringbuffer[p];
    //storageBarrier();
    workgroupBarrier();
    atomicStore(&queue.tickets[p], 2 * ((pos + QUEUE_CAPACITY) / QUEUE_CAPACITY));
    return element;
}

fn enqueue(LID: u32, element: u32) -> bool {
    //loop {
    for (var l: u32 = 0; l < LOOP_LIMIT; l++) {
        if (ensureEnqueue()) {
            break;
        }
        let head: u32 = atomicLoad(&queue.head);
        let tail: u32 = atomicLoad(&queue.tail);
        if (QUEUE_CAPACITY <= (tail - head) && (tail - head) < QUEUE_CAPACITY + (WORKGROUP_THREADS / 2)) {
            return false;
        }

        //storageBarrier();
        workgroupBarrier();
    }
    putData(element);
    return true;
}

fn dequeue() -> u32 {
    //loop {
    for (var l: u32 = 0; l < LOOP_LIMIT; l++) {
        if (ensureDequeue()) {
            break;
        }
        let head: u32 = atomicLoad(&queue.head);
        let tail: u32 = atomicLoad(&queue.tail);

        if (QUEUE_CAPACITY + (WORKGROUP_THREADS / 2) <= (tail - head) - 1) {
            return QUEUE_FAILURE;
        }

        //storageBarrier();
        workgroupBarrier();
    }
    return readData();
}
*/

/* DISPATCH */

override INPUT_SIZE: u32;

@group(0) @binding(0) var<storage, read_write> data: array<u32>;
@group(0) @binding(1) var<storage, read_write> debug: Queue;

@compute @workgroup_size(WORKGROUP_THREADS) fn cs(
    @builtin(local_invocation_index) LID: u32
) {
    /* INITIALIZE */

    if (LID == 0) {
        for (var i: u32 = 0; i < INPUT_SIZE / 2; i++) {
            enqueue(INPUT_SIZE + i);
            //atomicAdd(&queue.workLeft, 1);
        }
    }
    /*
    workgroupBarrier();

    /* PERSISTENT */

    for (var l: u32 = 0; l < LOOP_LIMIT; l++) {

        let current: u32 = dequeue();

        // try to get work
        if (current != QUEUE_FAILURE) {

            // do work
            let a: u32 = data[current * 2 - (INPUT_SIZE * 2)];
            let b: u32 = data[current * 2 - (INPUT_SIZE * 2) + 1];
            data[current] = a + b;

            // push new work
            // attention! this data lookup wont work because not atomic and 
            // no control barrier!
            /*
            if (current % 2 == 0 && data[current + 1] != 0) {
                let next: u32 = (current + (INPUT_SIZE * 2)) / 2;
                enqueue(next);
                //atomicAdd(&queue.workLeft, 1);
            }
            if (current % 2 == 1 && data[current - 1] != 0) {
                let next: u32 = (current - 1 + (INPUT_SIZE * 2)) / 2;
                enqueue(next);
                //atomicAdd(&queue.workLeft, 1);
            }
            */
            
            //atomicSub(&queue.workLeft, 1);

        // terminate if no dequeue and no work/none working
        } else /*if (atomicLoad(&queue.workLeft) == 0)*/ {
            break;
        }
    }
    */
    data[0] = data[0];

    /* DEBUG */

    if (LID == 0) {
        atomicStore(&debug.head, atomicLoad(&queue.head));
        atomicStore(&debug.tail, atomicLoad(&queue.tail));
        debug.ringbuffer = queue.ringbuffer;
    }
}