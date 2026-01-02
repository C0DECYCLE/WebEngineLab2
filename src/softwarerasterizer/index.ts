/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, January 2026
 */

import { int, float, Nullable, Undefinable } from "../utilities/utils.type.js";
import { assert, toRadian } from "../utilities/utils.js";
import { Vec3 } from "../utilities/Vec3.js";
import { Mat4 } from "../utilities/Mat4.js";
import { Controller } from "../Controller.js";
import { OBJParseResult } from "../OBJParser.js";
import { Stats } from "../Stats.js";
import { RollingAverage } from "../RollingAverage.js";
import { loadOBJ } from "../instancebatching/helper.js";

function createCanvas(): HTMLCanvasElement {
    const canvas: HTMLCanvasElement = document.createElement("canvas");
    canvas.width = document.body.clientWidth * devicePixelRatio;
    canvas.height = document.body.clientHeight * devicePixelRatio;
    canvas.style.position = "absolute";
    canvas.style.top = "0px";
    canvas.style.left = "0px";
    canvas.style.width = "100%";
    canvas.style.height = "100%";
    document.body.appendChild(canvas);
    return canvas;
}

//////////// SETUP ////////////

const canvas: HTMLCanvasElement = createCanvas();
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

//////////// SHADER ////////////

/*
const computeShader: GPUShaderModule = device.createShaderModule({
    code: await fetch("./shaders/meshoptimizer/compute.wgsl").then(
        async (response: Response) => await response.text(),
    ),
});
 */
const hardwareShader: GPUShaderModule = device.createShaderModule({
    code: await fetch("./shaders/softwarerasterizer/hardware.wgsl").then(
        async (response: Response) => await response.text(),
    ),
});
const combineShader: GPUShaderModule = device.createShaderModule({
    code: await fetch("./shaders/softwarerasterizer/combine.wgsl").then(
        async (response: Response) => await response.text(),
    ),
});

//////////// CONSTS ////////////

const byteSize: int = 4;
const instanceStride: int = 3 + 1;
const instanceCount: int = 16;
const spawnSize: int = 4;

//////////// GPU TIMING ////////////

const capacity: int = 6;
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

//////////// PIPELINE ////////////

/*
const computePipeline: GPUComputePipeline = device.createComputePipeline({
    layout: "auto",
    compute: {
        module: computeShader,
        entryPoint: "cs",
    },
});
 */
const hardwarePipeline: GPURenderPipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: {
        module: hardwareShader,
        entryPoint: "vs",
    },
    fragment: {
        module: hardwareShader,
        entryPoint: "fs",
        targets: [{ format: presentationFormat }],
        constants: {
            SCREEN_WIDTH: canvas.width,
            SCREEN_HEIGHT: canvas.height,
        },
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
        targets: [{ format: presentationFormat }],
        constants: {
            SCREEN_WIDTH: canvas.width,
            SCREEN_HEIGHT: canvas.height,
        },
    },
});

//////////// UNIFORM ////////////

const uniformFloats: int = 4 * 4;
const uniformData: Float32Array = new Float32Array(uniformFloats);
const uniformBuffer: GPUBuffer = device.createBuffer({
    size: uniformData.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});
device!.queue.writeBuffer(uniformBuffer, 0, uniformData.buffer);

//////////// GEOMETRY ////////////

const data: OBJParseResult = await loadOBJ("./resources/bunny.obj");
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
    new Vec3(
        i % spawnSize,
        Math.floor(i / (spawnSize * spawnSize)),
        Math.floor(i / spawnSize) % spawnSize,
    )
        .scale(3)
        .store(instancesData, i * instanceStride);
}
const instancesBuffer: GPUBuffer = device.createBuffer({
    size: instancesData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(instancesBuffer, 0, instancesData.buffer);

//////////// FRAME ////////////

const frameBuffer: GPUBuffer = device.createBuffer({
    size: canvas.width * canvas.height * byteSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});

//////////// BINDGROUP ////////////

/*
const computeBindGroup: GPUBindGroup = device.createBindGroup({
    layout: computePipeline.getBindGroupLayout(0),
    entries: [
        { binding: 0, resource: uniformBuffer },
    ],
});
*/
const hardwareBindGroup: GPUBindGroup = device.createBindGroup({
    layout: hardwarePipeline.getBindGroupLayout(0),
    entries: [
        { binding: 0, resource: uniformBuffer },
        { binding: 1, resource: vertexBuffer },
        { binding: 2, resource: instancesBuffer },
        { binding: 3, resource: frameBuffer },
    ],
});
const combineBindGroup: GPUBindGroup = device.createBindGroup({
    layout: combinePipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: frameBuffer }],
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
//stats.set("compute delta", 0);
stats.set("hardware delta", 0);
stats.set("combine delta", 0);
stats.show();
const frameDelta: RollingAverage = new RollingAverage(60);
//const computeDelta: RollingAverage = new RollingAverage(60);
const hardwareDelta: RollingAverage = new RollingAverage(60);
const combineDelta: RollingAverage = new RollingAverage(60);

//////////// RENDER FRAME ////////////

async function render(now: float): Promise<void> {
    assert(context && device);

    //////////// UPDATE ////////////

    control.update();
    cameraView.view(cameraPos, cameraDir, up);
    viewProjection.multiply(cameraView, projection).store(uniformData, 0);
    device.queue.writeBuffer(uniformBuffer, 0, uniformData.buffer);

    //////////// RENDER ////////////

    const target: GPUTextureView = context.getCurrentTexture().createView();

    const commandEncoder: GPUCommandEncoder = device.createCommandEncoder();

    commandEncoder.clearBuffer(frameBuffer);

    /*
    const computePass: GPUComputePassEncoder = renderEncoder.beginComputePass({
        timestampWrites: {
            querySet: querySet,
            beginningOfPassWriteIndex: 2,
            endOfPassWriteIndex: 3,
        },
    });
    computePass.setPipeline(computePipeline);
    computePass.setBindGroup(0, computeBindGroup);
    computePass.dispatchWorkgroups(instanceCount, 1, 1);
    computePass.end();
    */

    const hardwarePass: GPURenderPassEncoder = commandEncoder.beginRenderPass({
        colorAttachments: [
            {
                view: target,
                clearValue: [0, 0, 0, 0],
                loadOp: "clear",
                storeOp: "store",
            },
        ],
        timestampWrites: {
            querySet: querySet,
            beginningOfPassWriteIndex: 0,
            endOfPassWriteIndex: 1,
        },
    });
    hardwarePass.setPipeline(hardwarePipeline);
    hardwarePass.setBindGroup(0, hardwareBindGroup);
    hardwarePass.setIndexBuffer(indexBuffer, "uint32");
    hardwarePass.drawIndexed(data.indicesCount!, instanceCount);
    hardwarePass.end();

    const combinePass: GPURenderPassEncoder = commandEncoder.beginRenderPass({
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
            beginningOfPassWriteIndex: 4,
            endOfPassWriteIndex: 5,
        },
    });
    combinePass.setPipeline(combinePipeline);
    combinePass.setBindGroup(0, combineBindGroup);
    combinePass.draw(3);
    combinePass.end();

    commandEncoder.resolveQuerySet(querySet, 0, capacity, queryBuffer, 0);
    if (queryReadbackBuffer.mapState === "unmapped") {
        commandEncoder.copyBufferToBuffer(
            queryBuffer,
            0,
            queryReadbackBuffer,
            0,
            queryReadbackBuffer.size,
        );
    }

    device.queue.submit([commandEncoder.finish()]);

    if (queryReadbackBuffer.mapState === "unmapped") {
        queryReadbackBuffer.mapAsync(GPUMapMode.READ).then(() => {
            const timingsNanoseconds: BigInt64Array = new BigInt64Array(
                queryReadbackBuffer.getMappedRange().slice(0),
            );
            queryReadbackBuffer.unmap();
            /*
            stats.set(
                "compute delta",
                Number(timingsNanoseconds[3] - timingsNanoseconds[2]) /
                    1_000_000,
            );
            computeDelta.sample(stats.get("compute delta")!);
            */
            stats.set(
                "hardware delta",
                Number(timingsNanoseconds[1] - timingsNanoseconds[0]) /
                    1_000_000,
            );
            hardwareDelta.sample(stats.get("hardware delta")!);
            stats.set(
                "combine delta",
                Number(timingsNanoseconds[5] - timingsNanoseconds[4]) /
                    1_000_000,
            );
            combineDelta.sample(stats.get("combine delta")!);
        });
    }

    //////////// FRAME ////////////

    stats.set("frame delta", now - stats.get("frame delta")!);
    frameDelta.sample(stats.get("frame delta")!);
    // prettier-ignore
    stats.update(`
        <b>frame rate: ${(1_000 / frameDelta.get()).toFixed(0)} fps</b><br>
        frame delta: ${frameDelta.get().toFixed(2)} ms<br>
        <br>
        <b>hardware delta: ${hardwareDelta.get().toFixed(2)} ms<b><br>
        <b>combine delta: ${combineDelta.get().toFixed(2)} ms<b><br>
    `);
    stats.set("frame delta", now);

    requestAnimationFrame(render);
}
requestAnimationFrame(render);
