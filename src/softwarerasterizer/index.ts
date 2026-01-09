/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger
 */

import { Controller } from "../Controller.js";
import { loadOBJ } from "../instancebatching/helper.js";
import { OBJParseResult } from "../OBJParser.js";
import { RollingAverage } from "../RollingAverage.js";
import { Stats } from "../Stats.js";
import { Mat4 } from "../utilities/Mat4.js";
import { assert, toRadian } from "../utilities/utils.js";
import { float, int, Nullable, Undefinable } from "../utilities/utils.type.js";
import { Vec3 } from "../utilities/Vec3.js";
import { includeExternal, voxelizeOBJ } from "./helper.js";

//////////// CONSTS ////////////

const byteSize: int = 4;
const instanceStride: int = 3 + 1;
const instanceCount: int = 16;
const spawnGrid: int = 4;
const spawnScale: float = 4;
const voxelSize: float = 0.1;
const workgroupSize: int = 64;

//////////// SETUP ////////////

const canvas: HTMLCanvasElement = document.createElement("canvas");
canvas.width = document.body.clientWidth * devicePixelRatio;
canvas.height = document.body.clientHeight * devicePixelRatio;
canvas.style.position = "absolute";
canvas.style.top = "0px";
canvas.style.left = "0px";
canvas.style.width = "100%";
canvas.style.height = "100%";
document.body.appendChild(canvas);
const adapter: Nullable<GPUAdapter> = await navigator.gpu?.requestAdapter();
const device: Undefinable<GPUDevice> = await adapter?.requestDevice({
    requiredFeatures: ["timestamp-query"],
});
const context: Nullable<GPUCanvasContext> = canvas.getContext("webgpu");
if (!device || !context) {
    throw new Error("Browser doesn't support WebGPU.");
}
const presentationFormat: GPUTextureFormat =
    navigator.gpu.getPreferredCanvasFormat();
context.configure({
    device: device,
    format: presentationFormat,
});

//////////// CAMERA CONTROL ////////////

const cameraView: Mat4 = new Mat4();
const projection: Mat4 = Mat4.Perspective(
    60 * toRadian,
    canvas.width / canvas.height,
    0.01,
    1000,
);
const viewProjection: Mat4 = new Mat4();
const cameraPos: Vec3 = new Vec3(0, 1, 2);
const cameraDir: Vec3 = new Vec3(0, 0.5, 1).normalize();
const up: Vec3 = new Vec3(0, 1, 0);
const cameraData: Float32Array = new Float32Array(4 + 4 * 4);
const cameraBuffer: GPUBuffer = device.createBuffer({
    size: cameraData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(cameraBuffer, 0, cameraData.buffer);
const control: Controller = new Controller(canvas, {
    position: cameraPos,
    direction: cameraDir,
});

//////////// STATS ////////////

const deltaNames: string[] = [
    "cull",
    "software0",
    "software1",
    "hardware",
    "combine",
];
const deltas: Map<string, RollingAverage> = new Map<string, RollingAverage>();
const stats: Stats = new Stats();
deltas.set("frame", new RollingAverage(60));
stats.set("frame" + " delta", 0);
deltas.set("gpu", new RollingAverage(60));
stats.set("gpu" + " delta", 0);
for (const name of deltaNames) {
    deltas.set(name, new RollingAverage(60));
    stats.set(name + " delta", 0);
}
stats.show();

//////////// GEOMETRY ////////////

const data: OBJParseResult = await loadOBJ("./resources/house.obj");
assert(data.indices && data.indicesCount);
const voxelData: Float32Array = voxelizeOBJ(data, voxelSize);
const voxelBuffer: GPUBuffer = device.createBuffer({
    size: voxelData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(voxelBuffer, 0, voxelData.buffer);
const vertexBuffer: GPUBuffer = device.createBuffer({
    size: data.vertices.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(vertexBuffer, 0, data.vertices.buffer);
const indexBuffer: GPUBuffer = device.createBuffer({
    size: data.indices.byteLength,
    usage:
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.INDEX,
});
device.queue.writeBuffer(indexBuffer, 0, data.indices.buffer);

//////////// INSTANCES ////////////

const instancesData: Float32Array = new Float32Array(
    instanceCount * instanceStride,
);
for (let i: int = 0; i < instanceCount; i++) {
    let x: float = i % spawnGrid;
    let y: float = Math.floor(i / (spawnGrid * spawnGrid));
    let z: float = Math.floor(i / spawnGrid) % spawnGrid;
    let position: Vec3 = new Vec3(x, y, z).scale(spawnScale);
    position.store(instancesData, i * instanceStride);
}
const instancesBuffer: GPUBuffer = device.createBuffer({
    size: instancesData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(instancesBuffer, 0, instancesData.buffer);

//////////// SOFTWARE ////////////

const software0ArgsBuffer: GPUBuffer = device.createBuffer({
    size: 3 * byteSize,
    usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.INDIRECT,
});
const voxelCount: int = voxelData.length / 8;
device.queue.writeBuffer(
    software0ArgsBuffer,
    0,
    new Uint32Array([Math.ceil(voxelCount / workgroupSize), 0, 1]).buffer,
);
const software1ArgsBuffer: GPUBuffer = device.createBuffer({
    size: 3 * byteSize,
    usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.INDIRECT,
});
device.queue.writeBuffer(
    software1ArgsBuffer,
    0,
    new Uint32Array([0, 1, 1]).buffer,
);
const softwareInstancesBuffer: GPUBuffer = device.createBuffer({
    size: instanceCount * 1 * byteSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});
const cacheBuffer: GPUBuffer = device.createBuffer({
    size: 1024 * 64 * (3 + 1) * byteSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});

//////////// HARDWARE ////////////

const hardwareArgsBuffer: GPUBuffer = device.createBuffer({
    size: 5 * byteSize,
    usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.INDIRECT,
});
device.queue.writeBuffer(
    hardwareArgsBuffer,
    0,
    new Uint32Array([data.indicesCount, 0, 0, 0, 0]).buffer,
);
const hardwareInstancesBuffer: GPUBuffer = device.createBuffer({
    size: instanceCount * 1 * byteSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});

//////////// FRAME ////////////

const frameBuffer: GPUBuffer = device.createBuffer({
    size: canvas.width * canvas.height * byteSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});

//////////// GPU TIMING ////////////

const capacity: int = deltaNames.length * 2;
const querySet: GPUQuerySet = device.createQuerySet({
    type: "timestamp",
    count: capacity,
});
const queryBuffer: GPUBuffer = device.createBuffer({
    size: capacity * (byteSize * 2), //64bit
    usage:
        GPUBufferUsage.QUERY_RESOLVE |
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST,
});
const queryReadbackBuffer: GPUBuffer = device.createBuffer({
    size: queryBuffer.size,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
});

//////////// SHADER ////////////

const cullShader: GPUShaderModule = device.createShaderModule({
    code: await includeExternal("cull.wgsl"),
});
const software0Shader: GPUShaderModule = device.createShaderModule({
    code: await includeExternal("software0.wgsl"),
});
const software1Shader: GPUShaderModule = device.createShaderModule({
    code: await includeExternal("software1.wgsl"),
});
const hardwareShader: GPUShaderModule = device.createShaderModule({
    code: await includeExternal("hardware.wgsl"),
});
const combineShader: GPUShaderModule = device.createShaderModule({
    code: await includeExternal("combine.wgsl"),
});

//////////// PIPELINE ////////////

const cullPipeline: GPUComputePipeline = device.createComputePipeline({
    layout: "auto",
    compute: {
        module: cullShader,
        entryPoint: "cs",
        constants: {
            WORKGROUP_SIZE: workgroupSize,
        },
    },
});
const software0Pipeline: GPUComputePipeline = device.createComputePipeline({
    layout: "auto",
    compute: {
        module: software0Shader,
        entryPoint: "cs",
        constants: {
            SCREEN_WIDTH: canvas.width,
            SCREEN_HEIGHT: canvas.height,
            WORKGROUP_SIZE_X: workgroupSize,
        },
    },
});
const software1Pipeline: GPUComputePipeline = device.createComputePipeline({
    layout: "auto",
    compute: {
        module: software1Shader,
        entryPoint: "cs",
        constants: {
            SCREEN_WIDTH: canvas.width,
            SCREEN_HEIGHT: canvas.height,
        },
    },
});
const hardwarePipeline: GPURenderPipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: {
        module: hardwareShader,
        entryPoint: "vs",
    },
    fragment: {
        module: hardwareShader,
        entryPoint: "fs",
        constants: {
            SCREEN_WIDTH: canvas.width,
            SCREEN_HEIGHT: canvas.height,
        },
        targets: [{ format: presentationFormat }],
    },
    primitive: {
        cullMode: "back",
    },
});
const combinePipeline: GPURenderPipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: {
        module: combineShader,
        entryPoint: "vs",
    },
    fragment: {
        module: combineShader,
        entryPoint: "fs",
        constants: {
            SCREEN_WIDTH: canvas.width,
            SCREEN_HEIGHT: canvas.height,
        },
        targets: [{ format: presentationFormat }],
    },
});

//////////// BINDGROUP ////////////

const cullBindGroup: GPUBindGroup = device.createBindGroup({
    layout: cullPipeline.getBindGroupLayout(0),
    entries: [
        { binding: 0, resource: cameraBuffer },
        { binding: 1, resource: instancesBuffer },
        { binding: 2, resource: software0ArgsBuffer },
        { binding: 3, resource: softwareInstancesBuffer },
        { binding: 4, resource: hardwareArgsBuffer },
        { binding: 5, resource: hardwareInstancesBuffer },
    ],
});
const software0BindGroup: GPUBindGroup = device.createBindGroup({
    layout: software0Pipeline.getBindGroupLayout(0),
    entries: [
        { binding: 0, resource: cameraBuffer },
        { binding: 1, resource: voxelBuffer },
        { binding: 2, resource: instancesBuffer },
        { binding: 3, resource: softwareInstancesBuffer },
        { binding: 4, resource: software1ArgsBuffer },
        { binding: 5, resource: cacheBuffer },
    ],
});
const software1BindGroup: GPUBindGroup = device.createBindGroup({
    layout: software1Pipeline.getBindGroupLayout(0),
    entries: [
        { binding: 0, resource: cacheBuffer },
        { binding: 1, resource: frameBuffer },
    ],
});
const hardwareBindGroup: GPUBindGroup = device.createBindGroup({
    layout: hardwarePipeline.getBindGroupLayout(0),
    entries: [
        { binding: 0, resource: cameraBuffer },
        { binding: 1, resource: vertexBuffer },
        { binding: 2, resource: instancesBuffer },
        { binding: 3, resource: hardwareInstancesBuffer },
        { binding: 4, resource: frameBuffer },
    ],
});
const combineBindGroup: GPUBindGroup = device.createBindGroup({
    layout: combinePipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: frameBuffer }],
});

//////////// RENDER FRAME ////////////

function frame(now: float): void {
    assert(context && device && data.indices && data.indicesCount);

    //////////// UPDATE ////////////

    control.update();
    cameraPos.store(cameraData, 0);
    cameraView.view(cameraPos, cameraDir, up);
    viewProjection.multiply(cameraView, projection).store(cameraData, 4);
    device.queue.writeBuffer(cameraBuffer, 0, cameraData.buffer);

    //////////// ENCODE ////////////

    const target: GPUTextureView = context.getCurrentTexture().createView();

    const encoder: GPUCommandEncoder = device.createCommandEncoder();

    encoder.clearBuffer(software0ArgsBuffer, 1 * byteSize, 1 * byteSize);
    encoder.clearBuffer(software1ArgsBuffer, 0 * byteSize, 1 * byteSize);
    encoder.clearBuffer(hardwareArgsBuffer, 1 * byteSize, 1 * byteSize);
    encoder.clearBuffer(frameBuffer);

    const cullPass: GPUComputePassEncoder = encoder.beginComputePass({
        timestampWrites: {
            querySet: querySet,
            beginningOfPassWriteIndex: 0,
            endOfPassWriteIndex: 1,
        },
    });
    cullPass.setPipeline(cullPipeline);
    cullPass.setBindGroup(0, cullBindGroup);
    cullPass.dispatchWorkgroups(Math.ceil(instanceCount / workgroupSize));
    cullPass.end();

    const software0Pass: GPUComputePassEncoder = encoder.beginComputePass({
        timestampWrites: {
            querySet: querySet,
            beginningOfPassWriteIndex: 2,
            endOfPassWriteIndex: 3,
        },
    });
    software0Pass.setPipeline(software0Pipeline);
    software0Pass.setBindGroup(0, software0BindGroup);
    software0Pass.dispatchWorkgroupsIndirect(software0ArgsBuffer, 0);
    software0Pass.end();

    const software1Pass: GPUComputePassEncoder = encoder.beginComputePass({
        timestampWrites: {
            querySet: querySet,
            beginningOfPassWriteIndex: 4,
            endOfPassWriteIndex: 5,
        },
    });
    software1Pass.setPipeline(software1Pipeline);
    software1Pass.setBindGroup(0, software1BindGroup);
    software1Pass.dispatchWorkgroupsIndirect(software1ArgsBuffer, 0);
    software1Pass.end();

    const hardwarePass: GPURenderPassEncoder = encoder.beginRenderPass({
        colorAttachments: [
            {
                view: target,
                clearValue: [0, 0, 0, 0],
                loadOp: "clear",
                storeOp: "discard",
            },
        ],
        timestampWrites: {
            querySet: querySet,
            beginningOfPassWriteIndex: 6,
            endOfPassWriteIndex: 7,
        },
    });
    hardwarePass.setPipeline(hardwarePipeline);
    hardwarePass.setBindGroup(0, hardwareBindGroup);
    hardwarePass.setIndexBuffer(indexBuffer, "uint32");
    hardwarePass.drawIndexedIndirect(hardwareArgsBuffer, 0);
    hardwarePass.end();

    const combinePass: GPURenderPassEncoder = encoder.beginRenderPass({
        colorAttachments: [
            {
                view: target,
                clearValue: [1, 1, 1, 1],
                loadOp: "clear",
                storeOp: "store",
            },
        ],
        timestampWrites: {
            querySet: querySet,
            beginningOfPassWriteIndex: 8,
            endOfPassWriteIndex: 9,
        },
    });
    combinePass.setPipeline(combinePipeline);
    combinePass.setBindGroup(0, combineBindGroup);
    combinePass.draw(3);
    combinePass.end();

    encoder.resolveQuerySet(querySet, 0, capacity, queryBuffer, 0);
    if (queryReadbackBuffer.mapState === "unmapped") {
        encoder.copyBufferToBuffer(
            queryBuffer,
            0,
            queryReadbackBuffer,
            0,
            queryReadbackBuffer.size,
        );
    }

    device.queue.submit([encoder.finish()]);

    if (queryReadbackBuffer.mapState === "unmapped") {
        queryReadbackBuffer.mapAsync(GPUMapMode.READ).then(() => {
            const arrayBuffer: ArrayBuffer = queryReadbackBuffer
                .getMappedRange()
                .slice(0);
            queryReadbackBuffer.unmap();
            const timingsNanoseconds: BigInt64Array = new BigInt64Array(
                arrayBuffer,
            );
            const timings: float[] = Array.from(timingsNanoseconds).map(
                (value: bigint) => Number(value) / 1_000_000,
            );
            let min: float = Math.min(...timings);
            let max: float = Math.max(...timings);
            stats.set("gpu" + " delta", max - min);
            deltas.get("gpu")!.sample(stats.get("gpu" + " delta")!);
            for (let i: int = 0; i < deltaNames.length; i++) {
                const name: string = deltaNames[i];
                const a: float = timings[i * 2 + 0];
                const b: float = timings[i * 2 + 1];
                stats.set(name + " delta", b - a);
                deltas.get(name)!.sample(stats.get(name + " delta")!);
            }
        });
    }

    //////////// STATS ////////////

    stats.set("frame delta", now - stats.get("frame delta")!);
    deltas.get("frame")!.sample(stats.get("frame delta")!);
    // prettier-ignore
    stats.update(`
        <b>frame rate: ${(1_000 / deltas.get("frame")!.get()).toFixed(0)} fps</b><br>
        frame delta: ${deltas.get("frame")!.get().toFixed(2)} ms<br>
        gpu delta: ${deltas.get("gpu")!.get().toFixed(2)} ms<br>
        <br>
        <b>cull delta: ${deltas.get("cull")!.get().toFixed(2)} ms<b><br>
        <b>software0 delta: ${deltas.get("software0")!.get().toFixed(2)} ms<b><br>
        <b>software1 delta: ${deltas.get("software1")!.get().toFixed(2)} ms<b><br>
        <b>hardware delta: ${deltas.get("hardware")!.get().toFixed(2)} ms<b><br>
        <b>combine delta: ${deltas.get("combine")!.get().toFixed(2)} ms<b><br>
        `);
    stats.set("frame delta", now);

    requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
