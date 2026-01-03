/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, January 2026
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

//////////// CONSTS ////////////

const byteSize: int = 4;
const instanceStride: int = 3 + 1;
const instanceCount: int = 16;
const spawnSize: int = 4;

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
const control: Controller = new Controller(canvas, {
    position: cameraPos,
    direction: cameraDir,
});

//////////// STATS ////////////

const stats: Stats = new Stats();
stats.set("frame delta", 0);
stats.set("software0 delta", 0);
stats.set("software1 delta", 0);
stats.set("hardware delta", 0);
stats.set("combine delta", 0);
stats.show();
const frameDelta: RollingAverage = new RollingAverage(60);
const software0Delta: RollingAverage = new RollingAverage(60);
const software1Delta: RollingAverage = new RollingAverage(60);
const hardwareDelta: RollingAverage = new RollingAverage(60);
const combineDelta: RollingAverage = new RollingAverage(60);

//////////// UNIFORM ////////////

const cameraData: Float32Array = new Float32Array(4 * 4);
const cameraBuffer: GPUBuffer = device.createBuffer({
    size: cameraData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});
device!.queue.writeBuffer(cameraBuffer, 0, cameraData.buffer);

//////////// GEOMETRY ////////////

const data: OBJParseResult = await loadOBJ("./resources/book.obj");
const vertexBuffer: GPUBuffer = device.createBuffer({
    size: data.vertices.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(vertexBuffer, 0, data.vertices.buffer);
const indexBuffer: GPUBuffer = device.createBuffer({
    size: data.indices!.byteLength,
    usage:
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.INDEX,
});
device.queue.writeBuffer(indexBuffer, 0, data.indices!.buffer);

//////////// INSTANCES ////////////

const instancesData: Float32Array = new Float32Array(
    instanceCount * instanceStride,
);
for (let i: int = 0; i < instanceCount; i++) {
    let x: float = i % spawnSize;
    let y: float = Math.floor(i / (spawnSize * spawnSize));
    let z: float = Math.floor(i / spawnSize) % spawnSize;
    let position: Vec3 = new Vec3(x, y, z).scale(3);
    position.store(instancesData, i * instanceStride);
}
const instancesBuffer: GPUBuffer = device.createBuffer({
    size: instancesData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(instancesBuffer, 0, instancesData.buffer);

//////////// SOFTWARE ////////////

const cacheBuffer: GPUBuffer = device.createBuffer({
    size: 1024 * 64 * (3 + 1) * byteSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});
const argsBuffer: GPUBuffer = device.createBuffer({
    size: 3 * byteSize,
    usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.INDIRECT,
});
device.queue.writeBuffer(argsBuffer, 0, new Uint32Array([0, 1, 1]).buffer);

//////////// FRAME ////////////

const frameBuffer: GPUBuffer = device.createBuffer({
    size: canvas.width * canvas.height * byteSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});

//////////// GPU TIMING ////////////

const capacity: int = 8;
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

const directory: string = "./shaders/softwarerasterizer/";
const includesDirectory: string = "./shaders/softwarerasterizer/includes/";
const key: string = "#include";

async function includeExternal(file: string): Promise<string> {
    const source: string = await (await fetch(directory + file)).text();
    const lines: string[] = source.split("\n");
    for (let i: int = 0; i < lines.length; i++) {
        const line: string = lines[i];
        if (!line.startsWith(key)) {
            continue;
        }
        const subfile: string = line.split(key)[1].trim().split(";")[0];
        lines[i] = await (await fetch(includesDirectory + subfile)).text();
    }
    return lines.join("\n");
}

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

const software0Pipeline: GPUComputePipeline = device.createComputePipeline({
    layout: "auto",
    compute: {
        module: software0Shader,
        entryPoint: "cs",
        constants: {
            SCREEN_WIDTH: canvas.width,
            SCREEN_HEIGHT: canvas.height,
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

const software0BindGroup: GPUBindGroup = device.createBindGroup({
    layout: software0Pipeline.getBindGroupLayout(0),
    entries: [
        { binding: 0, resource: cameraBuffer },
        { binding: 1, resource: vertexBuffer },
        { binding: 2, resource: argsBuffer },
        { binding: 3, resource: cacheBuffer },
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
        { binding: 3, resource: frameBuffer },
    ],
});
const combineBindGroup: GPUBindGroup = device.createBindGroup({
    layout: combinePipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: frameBuffer }],
});

//////////// RENDER FRAME ////////////

function frame(now: float): void {
    assert(context && device);

    //////////// UPDATE ////////////

    control.update();
    cameraView.view(cameraPos, cameraDir, up);
    viewProjection.multiply(cameraView, projection).store(cameraData, 0);
    device.queue.writeBuffer(cameraBuffer, 0, cameraData.buffer);

    //////////// ENCODE ////////////

    const target: GPUTextureView = context.getCurrentTexture().createView();

    const encoder: GPUCommandEncoder = device.createCommandEncoder();

    encoder.clearBuffer(argsBuffer, 0, 1 * byteSize);
    encoder.clearBuffer(frameBuffer);

    const software0Pass: GPUComputePassEncoder = encoder.beginComputePass({
        timestampWrites: {
            querySet: querySet,
            beginningOfPassWriteIndex: 0,
            endOfPassWriteIndex: 1,
        },
    });
    software0Pass.setPipeline(software0Pipeline);
    software0Pass.setBindGroup(0, software0BindGroup);
    const num: int = Math.floor(data.verticesCount / 1);
    software0Pass.dispatchWorkgroups(Math.ceil(num / 64));
    software0Pass.end();

    const software1Pass: GPUComputePassEncoder = encoder.beginComputePass({
        timestampWrites: {
            querySet: querySet,
            beginningOfPassWriteIndex: 2,
            endOfPassWriteIndex: 3,
        },
    });
    software1Pass.setPipeline(software1Pipeline);
    software1Pass.setBindGroup(0, software1BindGroup);
    software1Pass.dispatchWorkgroupsIndirect(argsBuffer, 0);
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
            beginningOfPassWriteIndex: 4,
            endOfPassWriteIndex: 5,
        },
    });
    hardwarePass.setPipeline(hardwarePipeline);
    hardwarePass.setBindGroup(0, hardwareBindGroup);
    hardwarePass.setIndexBuffer(indexBuffer, "uint32");
    hardwarePass.drawIndexed(data.indicesCount!, instanceCount);
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
            beginningOfPassWriteIndex: 6,
            endOfPassWriteIndex: 7,
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
            const timingsNanoseconds: BigInt64Array = new BigInt64Array(
                queryReadbackBuffer.getMappedRange().slice(0),
            );
            queryReadbackBuffer.unmap();
            stats.set(
                "software0 delta",
                Number(timingsNanoseconds[1] - timingsNanoseconds[0]) /
                    1_000_000,
            );
            software0Delta.sample(stats.get("software0 delta")!);
            stats.set(
                "software1 delta",
                Number(timingsNanoseconds[3] - timingsNanoseconds[2]) /
                    1_000_000,
            );
            software1Delta.sample(stats.get("software1 delta")!);
            stats.set(
                "hardware delta",
                Number(timingsNanoseconds[5] - timingsNanoseconds[4]) /
                    1_000_000,
            );
            hardwareDelta.sample(stats.get("hardware delta")!);
            stats.set(
                "combine delta",
                Number(timingsNanoseconds[7] - timingsNanoseconds[6]) /
                    1_000_000,
            );
            combineDelta.sample(stats.get("combine delta")!);
        });
    }

    //////////// STATS ////////////

    stats.set("frame delta", now - stats.get("frame delta")!);
    frameDelta.sample(stats.get("frame delta")!);
    // prettier-ignore
    stats.update(`
        <b>frame rate: ${(1_000 / frameDelta.get()).toFixed(0)} fps</b><br>
        frame delta: ${frameDelta.get().toFixed(2)} ms<br>
        <br>
        <b>software0 delta: ${software0Delta.get().toFixed(2)} ms<b><br>
        <b>software1 delta: ${software1Delta.get().toFixed(2)} ms<b><br>
        <b>hardware delta: ${hardwareDelta.get().toFixed(2)} ms<b><br>
        <b>combine delta: ${combineDelta.get().toFixed(2)} ms<b><br>
    `);
    stats.set("frame delta", now);

    requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
