/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger
 */

import { log } from "../utilities/logger.js";
import { assert, loadText, sum } from "../utilities/utils.js";
import { float, int, Nullable } from "../utilities/utils.type.js";
import { ensure, output, populate, simulate, verify } from "./helper.js";

/* CONSTANTS */

export const InputSize: int = 8; //1024 * 1024;
export const InputMax: int = 10; //1000;
export const InputLayerCount: int = Math.log2(InputSize);
log("Input size", InputSize);
log("Input max", InputMax);
log("Input layer count", InputLayerCount);

export const Bytes32: int = 4;
export const Bytes64: int = 8;

export const QueryStride: int = 2;
export const QueryCapacity: int = 2;

export const WorkGroupSize1D: int = 64;

export const MsToNanos: int = 1_000_000;

/* CPU */

const inputData: Uint32Array = new Uint32Array(InputSize);
populate(inputData);
log("Input data", inputData);
const truth: int = sum([...inputData.slice(0, InputSize)]);
log("Truth", truth);

log("----------------------------------------------------");

const simulationData: Uint32Array = simulate(inputData);
ensure(simulationData, inputData, InputSize);
assert(simulationData[InputSize * 2 - 2] === truth);
log("Simulation data", simulationData);
output(simulationData, true);

log("----------------------------------------------------");

/* DEVICE */

const adapter: Nullable<GPUAdapter> = await navigator.gpu.requestAdapter();
assert(adapter);
const device: GPUDevice = await adapter.requestDevice({
    requiredFeatures: ["timestamp-query"],
});

/* BUFFERS */

const legacyDataBuffer: GPUBuffer = device.createBuffer({
    size: InputSize * 2 * Bytes32,
    usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(
    legacyDataBuffer,
    0 * Bytes32,
    inputData.buffer,
    0 * Bytes32,
    InputSize * Bytes32,
);
const legacyOperationBuffers: GPUBuffer[] = [];
for (let i: int = 0; i < InputLayerCount; i++) {
    const legacyOperationBuffer: GPUBuffer = device.createBuffer({
        size: 2 * Bytes32,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    let offset: int = 0;
    for (let o: int = 0; o < i + 1; o++) {
        offset += Math.pow(2, InputLayerCount - o);
    }
    const length: int = Math.pow(2, InputLayerCount - (i + 1));
    device.queue.writeBuffer(
        legacyOperationBuffer,
        0 * Bytes32,
        new Uint32Array([offset, length]).buffer,
        0 * Bytes32,
        2 * Bytes32,
    );
    legacyOperationBuffers.push(legacyOperationBuffer);
}
const legacyReadbackBuffer: GPUBuffer = device.createBuffer({
    size: InputSize * 2 * Bytes32,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
});

const persistentDataBuffer: GPUBuffer = device.createBuffer({
    size: InputSize * 2 * Bytes32,
    usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(
    persistentDataBuffer,
    0 * Bytes32,
    inputData.buffer,
    0 * Bytes32,
    InputSize * Bytes32,
);
const persistentOperationBuffer: GPUBuffer = device.createBuffer({
    size: 1 * Bytes32,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(
    persistentOperationBuffer,
    0 * Bytes32,
    new Uint32Array([InputSize / 2]).buffer,
    0 * Bytes32,
    1 * Bytes32,
);
const persistentQueueBuffer: GPUBuffer = device.createBuffer({
    size: (65536 * 2 + 3) * Bytes32,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(
    persistentQueueBuffer,
    65536 * 2 * Bytes32,
    new Uint32Array([0, 1, 1]).buffer,
    0 * Bytes32,
    3 * Bytes32,
);
const persistentReadbackBuffer: GPUBuffer = device.createBuffer({
    size: InputSize * 2 * Bytes32,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
});

/* QUERIES */

const querySet: GPUQuerySet = device.createQuerySet({
    type: "timestamp",
    count: QueryCapacity * QueryStride,
});
const queryBuffer: GPUBuffer = device.createBuffer({
    size: QueryCapacity * QueryStride * Bytes64,
    usage:
        GPUBufferUsage.QUERY_RESOLVE |
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST,
});
const queryReadbackBuffer: GPUBuffer = device.createBuffer({
    size: QueryCapacity * QueryStride * Bytes64,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
});

/* SHADERS */

const legacyShader: GPUShaderModule = device.createShaderModule({
    code: await loadText("./shaders/persistentthreads/legacy.wgsl"),
});

const persistentShader: GPUShaderModule = device.createShaderModule({
    code: await loadText("./shaders/persistentthreads/persistent.wgsl"),
});

/* PIPELINES */

const legacyPipeline: GPUComputePipeline =
    await device.createComputePipelineAsync({
        layout: "auto",
        compute: {
            module: legacyShader,
            entryPoint: "cs",
            constants: {
                WORKGROUP_SIZE_1D: WorkGroupSize1D,
            },
        },
    });

const persistentPipeline: GPUComputePipeline =
    await device.createComputePipelineAsync({
        layout: "auto",
        compute: {
            module: persistentShader,
            entryPoint: "cs",
            constants: {
                WORKGROUP_SIZE_1D: WorkGroupSize1D,
            },
        },
    });

/* BINDGROUPS */

const legacyDataBindGroup: GPUBindGroup = device.createBindGroup({
    layout: legacyPipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: legacyDataBuffer }],
});
const legacyOperationBindGroups: GPUBindGroup[] = [];
for (let i: int = 0; i < InputLayerCount; i++) {
    const legacyOperationBindGroup: GPUBindGroup = device.createBindGroup({
        layout: legacyPipeline.getBindGroupLayout(1),
        entries: [{ binding: 0, resource: legacyOperationBuffers[i] }],
    });
    legacyOperationBindGroups.push(legacyOperationBindGroup);
}

const persistentBindGroup: GPUBindGroup = device.createBindGroup({
    layout: persistentPipeline.getBindGroupLayout(0),
    entries: [
        { binding: 0, resource: persistentDataBuffer },
        { binding: 1, resource: persistentOperationBuffer },
        { binding: 2, resource: persistentQueueBuffer },
    ],
});

/* ENCODER */

const encoder: GPUCommandEncoder = device.createCommandEncoder({});

const legacyPass: GPUComputePassEncoder = encoder.beginComputePass({
    timestampWrites: {
        querySet: querySet,
        beginningOfPassWriteIndex: 0 * QueryStride + 0,
        endOfPassWriteIndex: 0 * QueryStride + 1,
    },
});
legacyPass.setPipeline(legacyPipeline);
legacyPass.setBindGroup(0, legacyDataBindGroup);
for (let i: int = 0; i < InputLayerCount; i++) {
    legacyPass.setBindGroup(1, legacyOperationBindGroups[i]);
    const length: int = Math.pow(2, InputLayerCount - (i + 1));
    const count: int = Math.ceil(length / WorkGroupSize1D);
    legacyPass.dispatchWorkgroups(Math.max(1, count));
}
legacyPass.end();

encoder.copyBufferToBuffer(
    legacyDataBuffer,
    0 * Bytes32,
    legacyReadbackBuffer,
    0 * Bytes32,
    InputSize * 2 * Bytes32,
);

const persistentPass: GPUComputePassEncoder = encoder.beginComputePass({
    timestampWrites: {
        querySet: querySet,
        beginningOfPassWriteIndex: 1 * QueryStride + 0,
        endOfPassWriteIndex: 1 * QueryStride + 1,
    },
});
persistentPass.setPipeline(persistentPipeline);
persistentPass.setBindGroup(0, persistentBindGroup);
persistentPass.dispatchWorkgroups(Math.max(1, Math.ceil(1 / WorkGroupSize1D)));
persistentPass.end();

encoder.copyBufferToBuffer(
    persistentDataBuffer,
    0 * Bytes32,
    persistentReadbackBuffer,
    0 * Bytes32,
    InputSize * 2 * Bytes32,
);

encoder.resolveQuerySet(
    querySet,
    0 * QueryStride,
    QueryCapacity * QueryStride,
    queryBuffer,
    0 * Bytes64,
);
encoder.copyBufferToBuffer(
    queryBuffer,
    0 * Bytes64,
    queryReadbackBuffer,
    0 * Bytes64,
    QueryCapacity * QueryStride * Bytes64,
);

device.queue.submit([encoder.finish()]);

/* READBACK */

await queryReadbackBuffer.mapAsync(GPUMapMode.READ);
const queryData: BigInt64Array = new BigInt64Array(
    queryReadbackBuffer.getMappedRange(),
);

const pre0: float = Number(queryData[0 * QueryStride + 0]) / MsToNanos;
const post0: float = Number(queryData[0 * QueryStride + 1]) / MsToNanos;
log("Legacy GPU time (ms)", post0 - pre0);

await legacyReadbackBuffer.mapAsync(GPUMapMode.READ);
const legacyData: Uint32Array = new Uint32Array(
    legacyReadbackBuffer.getMappedRange(),
);
verify(legacyData, simulationData);
log("Legacy data", legacyData);
output(legacyData, true);

log("----------------------------------------------------");

const pre1: float = Number(queryData[1 * QueryStride + 0]) / MsToNanos;
const post1: float = Number(queryData[1 * QueryStride + 1]) / MsToNanos;
log("Persistent GPU time (ms)", post1 - pre1);

await persistentReadbackBuffer.mapAsync(GPUMapMode.READ);
const persistentData: Uint32Array = new Uint32Array(
    persistentReadbackBuffer.getMappedRange(),
);
//verify(persistentData, simulationData);
log("Persistent data", persistentData);
//output(persistentData, true);

log("----------------------------------------------------");
