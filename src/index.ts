/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, October 2023
 */

import {
    int,
    float,
    Nullable,
    Undefinable,
} from "../types/utilities/utils.type.js";
import { toRadian } from "./utilities/utils.js";
import { log } from "./utilities/logger.js";
import { Vec3 } from "./utilities/Vec3.js";
import { Mat4 } from "./utilities/Mat4.js";
import { Controller } from "./Controller.js";
import { OBJParser } from "./OBJParser.js";
import { Stats } from "./Stats.js";

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
const device: Undefinable<GPUDevice> = await adapter?.requestDevice();
const context: Nullable<GPUCanvasContext> = canvas.getContext("webgpu");
if (!device || !context) {
    throw new Error("SpaceEngine: Browser doesn't support WebGPU.");
}
const presentationFormat: GPUTextureFormat =
    navigator.gpu.getPreferredCanvasFormat();
context.configure({
    device: device,
    format: presentationFormat,
} as GPUCanvasConfiguration);

//////////// SHADER ////////////

const module: GPUShaderModule = device.createShaderModule({
    label: "normal shader",
    code: await fetch("./shaders/shader.wgsl").then(
        async (response: Response) => await response.text(),
    ),
} as GPUShaderModuleDescriptor);

//////////// PIPELINE ////////////

const pipeline: GPURenderPipeline = device.createRenderPipeline({
    label: "geometry pipeline",
    layout: "auto",
    vertex: {
        module,
        entryPoint: "vs",
    } as GPUVertexState,
    fragment: {
        module,
        entryPoint: "fs",
        targets: [{ format: presentationFormat }],
    } as GPUFragmentState,
    primitive: {
        cullMode: "back",
    } as GPUPrimitiveState,
    depthStencil: {
        depthWriteEnabled: true,
        depthCompare: "less",
        format: "depth24plus",
    } as GPUDepthStencilState,
} as GPURenderPipelineDescriptor);

//////////// RENDERPASS ////////////

const colorAttachment: GPURenderPassColorAttachment = {
    view: context!.getCurrentTexture().createView(),
    clearValue: [0.3, 0.3, 0.3, 1.0],
    loadOp: "clear",
    storeOp: "store",
} as GPURenderPassColorAttachment;

const depthTexture = device.createTexture({
    label: "depth texture",
    size: [canvas.width, canvas.height],
    format: "depth24plus",
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
});
const depthStencilAttachment: GPURenderPassDepthStencilAttachment = {
    view: depthTexture.createView(),
    depthClearValue: 1.0,
    depthLoadOp: "clear",
    depthStoreOp: "store",
} as GPURenderPassDepthStencilAttachment;

const renderPassDescriptor: GPURenderPassDescriptor = {
    label: "canvas render pass",
    colorAttachments: [colorAttachment],
    depthStencilAttachment: depthStencilAttachment,
} as GPURenderPassDescriptor;

//////////// UNIFORM ////////////

const byteSize: int = 4;
const uniformBufferSize: int = 16 * byteSize + 1 * byteSize;

const uniformBuffer: GPUBuffer = device.createBuffer({
    label: "uniforms uniform buffer",
    size:
        uniformBufferSize +
        (4 * byteSize - (uniformBufferSize % (4 * byteSize))),
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
} as GPUBufferDescriptor);

const uniformArrayBuffer = new ArrayBuffer(uniformBufferSize);
const floatValues: Float32Array = new Float32Array(uniformArrayBuffer);
const intValues: Uint32Array = new Uint32Array(uniformArrayBuffer);

const matrixOffset: int = 0;
const modeOffset: int = 16;

(window as any).setMode = (mode: 0 | 1 | 2) => {
    intValues[modeOffset] = mode;
    device?.queue.writeBuffer(
        uniformBuffer,
        modeOffset * byteSize,
        uniformArrayBuffer,
        modeOffset * byteSize,
    );
};
(window as any).setMode(1);

//////////// VERTECIES ////////////

const raw: string = await fetch("./resources/bunny.obj").then(
    async (response: Response) => await response.text(),
);
const parser: OBJParser = new OBJParser();
const vertexData: Float32Array = parser.parse(raw);
const vertexCount: int = vertexData.length / 4;

const vertexArrayBuffer = vertexData.buffer;

const verteciesBuffer: GPUBuffer = device.createBuffer({
    label: "vertices storage buffer",
    size: vertexArrayBuffer.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
} as GPUBufferDescriptor);

log(vertexCount);
/*
let diffX: int;
let diffY: int;
let diffZ: int;
function distanceSqrt(a: int, b: int): float {
    diffX = vertexData[b * 4 + 0] - vertexData[a * 4 + 0];
    diffY = vertexData[b * 4 + 1] - vertexData[a * 4 + 1];
    diffZ = vertexData[b * 4 + 2] - vertexData[a * 4 + 2];
    return diffX * diffX + diffY * diffY + diffZ * diffZ;
}

const clusterIdData: Uint32Array = new Uint32Array(vertexArrayBuffer);
let currentId: int = 1;
const clusters: Map<int, int[]> = new Map<int, int[]>();
function split(list: int[]): void {
    let i: int;
    if (list.length < 2048) {
        for (i = 0; i < list.length; i++) {
            clusterIdData[list[i] * 4 + 3] = currentId;
        }
        //log(currentId, list.length);
        clusters.set(currentId, list);
        currentId++;
        return;
    }
    const a: int[] = [];
    const b: int[] = [];
    const northpole: int = list[Math.floor(Math.random() * list.length)];
    let southpole: int = -1;
    let southpoleDistance: float = 0.0;
    let currentDistance: float;
    for (i = 0; i < list.length; i++) {
        if (list[i] === northpole) {
            continue;
        }
        currentDistance = distanceSqrt(list[i], northpole);
        if (southpole === -1 || currentDistance > southpoleDistance) {
            southpole = list[i];
            southpoleDistance = currentDistance;
        }
    }
    let aDistance: float;
    let bDistance: float;
    for (i = 0; i < list.length; i++) {
        if (list[i] === northpole) {
            a.push(list[i]);
            continue;
        }
        if (list[i] === southpole) {
            b.push(list[i]);
            continue;
        }
        aDistance = distanceSqrt(list[i], northpole);
        bDistance = distanceSqrt(list[i], southpole);
        if (aDistance < bDistance) {
            a.push(list[i]);
        } else {
            b.push(list[i]);
        }
    }
    split(a);
    split(b);
}
const initial: int[] = [];
for (let i: int = 0; i < vertexCount; i++) {
    initial.push(i);
    clusterIdData[i * 4 + 3] = 0;
}
const pre: float = performance.now();
split(initial);
log(clusters);
log("complete", currentId, performance.now() - pre);
*/

device.queue.writeBuffer(verteciesBuffer, 0, vertexArrayBuffer);

//////////// BINDGROUP ////////////

const bindGroup: GPUBindGroup = device.createBindGroup({
    label: "bind group for geometry",
    layout: pipeline.getBindGroupLayout(0),
    entries: [
        {
            binding: 0,
            resource: { buffer: uniformBuffer } as GPUBindingResource,
        } as GPUBindGroupEntry,
        {
            binding: 1,
            resource: { buffer: verteciesBuffer } as GPUBindingResource,
        } as GPUBindGroupEntry,
    ],
} as GPUBindGroupDescriptor);

//////////// MATRIX ////////////

const world: Mat4 = new Mat4();
const view: Mat4 = new Mat4();
const projection: Mat4 = Mat4.Perspective(
    60 * toRadian,
    canvas.width / canvas.height,
    0.01,
    1000.0,
);
const worldViewProjection: Mat4 = new Mat4();

const cameraPos: Vec3 = new Vec3(0.0, 2.0, 5.0);
const cameraDir: Vec3 = new Vec3(0.0, 0.0, 1.0);
const up: Vec3 = new Vec3(0.0, 1.0, 0.0);

//////////// CONTROL ////////////

const control: Controller = new Controller({
    position: cameraPos,
    direction: cameraDir,
});

canvas.addEventListener("click", () => {
    canvas.requestPointerLock();
});

//////////// STATS ////////////

const stats: Stats = new Stats();
stats.set("frame delta", 0);
stats.show();

async function render(now: float): Promise<void> {
    stats.time("cpu delta");

    //////////// UPDATE ////////////
    control.update();

    view.view(cameraPos, cameraDir, up);
    worldViewProjection
        .multiply(view, projection)
        .multiply(world, worldViewProjection)
        .store(floatValues, matrixOffset);
    device?.queue.writeBuffer(
        uniformBuffer,
        matrixOffset * byteSize,
        uniformArrayBuffer,
        matrixOffset * byteSize,
    );

    //////////// DRAW ////////////

    colorAttachment.view = context!.getCurrentTexture().createView();
    depthStencilAttachment.view = depthTexture.createView();

    const encoder: GPUCommandEncoder = device!.createCommandEncoder({
        label: "frame command encoder",
    } as GPUObjectDescriptorBase);

    const pass: GPURenderPassEncoder =
        encoder.beginRenderPass(renderPassDescriptor);
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(vertexCount);
    pass.end();

    const commandBuffer: GPUCommandBuffer = encoder.finish();
    device?.queue.submit([commandBuffer]);

    //////////// FRAME ////////////

    stats.set("frame delta", now - stats.get("frame delta")!);
    stats.time("cpu delta", "cpu delta");
    // prettier-ignore
    stats.update(`
            <b>frame rate: ${(1_000 / stats.get("frame delta")!).toFixed(1)} fps</b><br>
            frame delta: ${stats.get("frame delta")!.toFixed(2)} ms<br>
            <br>
            <b>cpu rate: ${(1_000 / stats.get("cpu delta")!).toFixed(1)} fps</b><br>
            cpu delta: ${stats.get("cpu delta")!.toFixed(2)} ms`
        );
    stats.set("frame delta", now);

    requestAnimationFrame(render);
}
requestAnimationFrame(render);
