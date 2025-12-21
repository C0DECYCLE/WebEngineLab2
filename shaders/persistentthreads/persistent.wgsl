const QUEUE_SIZE: u32 = 1024;
const QUEUE_MASK: u32 = QUEUE_SIZE - 1;
const QUEUE_FAILURE: u32 = 4294967295;

struct Queue {
    head: atomic<u32>,
    tail: atomic<u32>,
    activeCount: atomic<u32>,
    data: array<u32, QUEUE_SIZE>,
}

fn enqueue(value: u32) -> bool {
    let h: u32 = atomicLoad(&queue.head);
    let t: u32 = atomicLoad(&queue.tail);
    if (h - t >= QUEUE_SIZE) { 
        return false; 
    }
    let res = atomicCompareExchangeWeak(&queue.head, h, h + 1);
    if (!res.exchanged) {
        return false;
    }
    queue.data[h & QUEUE_MASK] = value;
    atomicAdd(&queue.activeCount, 1);
    return true;
}

fn dequeue(value: ptr<function, u32>) -> bool {
    if (atomicLoad(&queue.activeCount) == 0) {
        return false;
    }
    let t: u32 = atomicLoad(&queue.tail);
    let h: u32 = atomicLoad(&queue.head);
    if (t >= h) {
        return false;
    }
    let res = atomicCompareExchangeWeak(&queue.tail, t, t + 1);
    if (!res.exchanged) {
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
    @builtin(num_workgroups) numWorkgroups: vec3u,
    @builtin(global_invocation_id) globalInvocationId: vec3u
) {
    let workgroupCount: u32 = numWorkgroups.x;
    let index: u32 = globalInvocationId.x;

    // enqueue initial tasks
    if (workgroupCount == 1) {
        if (index == 0) {
            for (var i: u32 = 0; i < INPUT_SIZE / 2; i++) {
                enqueue(INPUT_SIZE + i);
            }
        }
        return;
    }

    // persistent thread
    for (var l: u32 = 0; l < QUEUE_FAILURE; l++) {

        var current: u32;

        // try to get work
        if (dequeue(&current)) {

            // do work
            let a: u32 = data[current * 2 - (INPUT_SIZE * 2)];
            let b: u32 = data[current * 2 - (INPUT_SIZE * 2) + 1];
            data[current] = a + b;

            // push new work
            if (current % 2 == 0 && data[current + 1] != 0) {
                let next: u32 = (current + (INPUT_SIZE * 2)) / 2;

                for (var k: u32 = 0; k < QUEUE_FAILURE; k++) {
                    if (enqueue(next)) {
                        break;
                    }
                }
            }
            if (current % 2 == 1 && data[current - 1] != 0) {
                let next: u32 = (current - 1 + (INPUT_SIZE * 2)) / 2;

                for (var k: u32 = 0; k < QUEUE_FAILURE; k++) {
                    if (enqueue(next)) {
                        break;
                    }
                }
            }
            
            atomicSub(&queue.activeCount, 1);
            
        } else {
    
            // terminate if no work and none working
            if (atomicLoad(&queue.activeCount) == 0) {
                break;
            }
            
            // optional small yield/spin to reduce contention
            var dummy: u32 = 0;
            for (var i: u32 = 0; i < 100; i++) {
                dummy += i;
            }
            if (dummy == 0xFFFFFFFFu) { 
                atomicAdd(&queue.activeCount, 0); 
            }
        }
    }
}