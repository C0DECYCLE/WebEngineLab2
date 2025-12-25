const THREAD_COUNT: u32 = 16384 * 64;

const QUEUE_SIZE: u32 = 65536;
const QUEUE_FAILURE: u32 = 4294967295;

const LOOP_LIMIT: u32 = 1024;

struct Queue {
    head: atomic<u32>,
    tail: atomic<u32>,
    count: atomic<i32>,
    ringbuffer: array<u32/*atomic<u32>*/, QUEUE_SIZE>,
    tickets: array<atomic<u32>, QUEUE_SIZE>,
    //
    workLeft: atomic<u32>,
}

fn ensureEnqueue() -> bool {
    var num: i32 = atomicLoad(&queue.count);
    var ensurance: bool = false;

    //loop {
    for (var l: u32 = 0; l < LOOP_LIMIT; l++) {
        if (ensurance || num >= i32(QUEUE_SIZE)) {
            break;
        }
        ensurance = atomicAdd(&queue.count, 1) < i32(QUEUE_SIZE);
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

        if (atomicLoad(&queue.tickets[pos]) == expected_ticket) {
            break;
        }
    }
}

fn putData(element: u32) {
    let pos: u32 = atomicAdd(&queue.tail, 1);
    let p: u32 = pos % QUEUE_SIZE;
    let expected: u32 = 2 * (pos / QUEUE_SIZE);
    waitForTicket(p, expected);
    queue.ringbuffer[p] = element;
    //storageBarrier();
    atomicStore(&queue.tickets[p], expected + 1);
}

fn readData() -> u32 {
    let pos: u32 = atomicAdd(&queue.head, 1);
    let p: u32 = pos % QUEUE_SIZE;
    let expected: u32 = 2 * (pos / QUEUE_SIZE);
    waitForTicket(p, expected + 1);
    var element: u32 = queue.ringbuffer[p];
    //storageBarrier();
    atomicStore(&queue.tickets[p], 2 * ((pos + QUEUE_SIZE) / QUEUE_SIZE));
    return element;
}

fn enqueue(element: u32) -> bool {
    //loop {
    for (var l: u32 = 0; l < LOOP_LIMIT; l++) {
        if (ensureEnqueue()) {
            break;
        }
        let head: u32 = atomicLoad(&queue.head);
        let tail: u32 = atomicLoad(&queue.tail);
        if (QUEUE_SIZE <= (tail - head) && (tail - head) < QUEUE_SIZE + (THREAD_COUNT / 2)) {
            return false;
        }

        //storageBarrier();
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

        if (QUEUE_SIZE + (THREAD_COUNT / 2) <= (tail - head) - 1) {
            return QUEUE_FAILURE;
        }

        //storageBarrier();
    }
    return readData();
}

override WORKGROUP_SIZE_1D: u32;
override INPUT_SIZE: u32;

@group(0) @binding(0) var<storage, read_write> data: array<u32>;
@group(0) @binding(1) var<storage, read_write> queue: Queue;

@compute @workgroup_size(WORKGROUP_SIZE_1D) fn cs(
    @builtin(num_workgroups) numWorkgroups: vec3u,
    @builtin(global_invocation_id) globalInvocationId: vec3u
) {
    let workgroupCount: u32 = numWorkgroups.x;
    let index: u32 = globalInvocationId.x;

    // initial tasks dispatch
    if (workgroupCount == 1) {
        if (index == 0) {
            for (var i: u32 = 0; i < INPUT_SIZE / 2; i++) {
                enqueue(INPUT_SIZE + i);
                atomicAdd(&queue.workLeft, 1);
            }
        }
        return;
    }

    // persistent thread dispatch
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
            if (current % 2 == 0 && data[current + 1] != 0) {
                let next: u32 = (current + (INPUT_SIZE * 2)) / 2;
                enqueue(next);
                atomicAdd(&queue.workLeft, 1);
            }
            if (current % 2 == 1 && data[current - 1] != 0) {
                let next: u32 = (current - 1 + (INPUT_SIZE * 2)) / 2;
                enqueue(next);
                atomicAdd(&queue.workLeft, 1);
            }
            
            atomicSub(&queue.workLeft, 1);

        // terminate if no dequeue and no work/none working
        } else if (atomicLoad(&queue.workLeft) == 0) {
            break;
        }
    }
}