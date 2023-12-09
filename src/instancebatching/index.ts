/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, October 2023
 */

import { int, float } from "../../types/utilities/utils.type.js";
import { Controller } from "../Controller.js";
import { Stats } from "../Stats.js";
import { RollingAverage } from "../RollingAverage.js";
import { createCanvas, loadOBJ } from "./helper.js";
import { Camera } from "./Camera.js";
import { GPUTiming } from "./GPUTiming.js";
import { OBJParseResult } from "../OBJParser.js";
import { log } from "../utilities/logger.js";
import { Vec3 } from "../utilities/Vec3.js";
import { dotit } from "../utilities/utils.js";

//////////// SETUP GPU ////////////

const canvas: HTMLCanvasElement = createCanvas();
const adapter: GPUAdapter = (await navigator.gpu?.requestAdapter())!;
const device: GPUDevice = (await adapter?.requestDevice({
    requiredFeatures: ["timestamp-query", "indirect-first-instance"],
} as GPUDeviceDescriptor))!;
const context: GPUCanvasContext = canvas.getContext("webgpu")!;
const presentationFormat: GPUTextureFormat =
    navigator.gpu.getPreferredCanvasFormat();
context.configure({
    device: device,
    format: presentationFormat,
} as GPUCanvasConfiguration);

//////////// CREATE CAMERA AND CONTROL ////////////

const camera: Camera = new Camera(canvas.width / canvas.height, 1000);
const control: Controller = new Controller(canvas, camera);

//////////// CREATE STATS ////////////

const stats: Stats = new Stats();
stats.set("frame delta", 0);
stats.set("gpu delta", 0);
stats.show();

const frameDelta: RollingAverage = new RollingAverage(60);
const cpuDelta: RollingAverage = new RollingAverage(60);
const gpuDelta: RollingAverage = new RollingAverage(60);

const gpuTiming: GPUTiming = new GPUTiming(device);

//////////// LOAD OBJ ////////////

const cube: OBJParseResult = await loadOBJ("./resources/cube.obj");
const icosphere: OBJParseResult = await loadOBJ("./resources/icosphere.obj");
const torus: OBJParseResult = await loadOBJ("./resources/torus.obj");
const cylinder: OBJParseResult = await loadOBJ("./resources/cylinder.obj");
const cone: OBJParseResult = await loadOBJ("./resources/cone.obj");
const suzanne: OBJParseResult = await loadOBJ("./resources/suzanne.obj");

const geometries: OBJParseResult[] = [
    cube,
    icosphere,
    torus,
    cylinder,
    cone,
    suzanne,
];

let bytes: int = 0;

//////////// SETUP UNIFORM ////////////

const uniformData: Float32Array = new Float32Array(4 * 4);
const uniformBuffer: GPUBuffer = device.createBuffer({
    label: "uniforms uniform buffer",
    size: uniformData.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
} as GPUBufferDescriptor);
bytes += uniformBuffer.size;

//////////// SETUP VERTICES ////////////

const vertexData: Float32Array = new Float32Array(
    geometries.flatMap((geometry: OBJParseResult) => [...geometry.positions]),
);
const vertexBuffer: GPUBuffer = device.createBuffer({
    label: "vertex buffer",
    size: vertexData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
} as GPUBufferDescriptor);
device.queue.writeBuffer(vertexBuffer, 0, vertexData);
log("vertices", vertexData.length / 4);
bytes += vertexBuffer.size;

//////////// SETUP INDICES ////////////

const indexData: Uint32Array = new Uint32Array(
    geometries.flatMap((geometry: OBJParseResult) => [...geometry.indices!]),
);
const indexBuffer: GPUBuffer = device.createBuffer({
    label: "index buffer",
    size: indexData.byteLength,
    usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
} as GPUBufferDescriptor);
device.queue.writeBuffer(indexBuffer, 0, indexData);
log("indices", indexData.length);
bytes += indexBuffer.size;

//////////// SETUP INSTANCES ////////////

const n: int = 10_000;
const count: int = n * geometries.length;
const attr: int = 3 + 1;
const floats: int = attr + attr;
const instanceData: Float32Array = new Float32Array(floats * count);
for (let i: int = 0; i < count; i++) {
    new Vec3(Math.random(), Math.random(), Math.random())
        .sub(0.5)
        .scale(Math.cbrt(n) * 10)
        .store(instanceData, i * floats);
    const obj: int = Math.floor(i / n) + 1;
    new Vec3(
        (obj * 345.323) % 1,
        (obj * 486.116) % 1,
        (obj * 193.735) % 1,
    ).store(instanceData, i * floats + attr);
}
const instanceBuffer: GPUBuffer = device.createBuffer({
    label: "instance buffer",
    size: instanceData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
} as GPUBufferDescriptor);
device.queue.writeBuffer(instanceBuffer, 0, instanceData);
log("instances", dotit(n), dotit(count));
bytes += instanceBuffer.size;

//////////// SETUP INDIRECTS ////////////

const indirectData: Uint32Array = new Uint32Array(5 * geometries.length);
let totalIndices: int = 0;
let totalPositions: int = 0;
geometries.forEach((geometry: OBJParseResult, i: int) => {
    indirectData.set(
        [geometry.indicesCount!, n, totalIndices, totalPositions, n * i],
        5 * i,
    );
    totalIndices += geometry.indicesCount!;
    totalPositions += geometry.positionsCount;
});
const indirectBuffer: GPUBuffer = device.createBuffer({
    label: "indirect buffer",
    size: indirectData.byteLength,
    usage:
        GPUBufferUsage.INDIRECT |
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST,
} as GPUBufferDescriptor);
device.queue.writeBuffer(indirectBuffer, 0, indirectData);
bytes += indirectBuffer.size;

log("vram", dotit(bytes));

//////////// LOAD SHADER ////////////

const shader: GPUShaderModule = device.createShaderModule({
    label: "shader",
    code: await fetch("./shaders/instancebatching/shader.wgsl").then(
        async (response: Response) => await response.text(),
    ),
} as GPUShaderModuleDescriptor);

//////////// CREATE PIPELINE ////////////

const pipeline: GPURenderPipeline = await device.createRenderPipelineAsync({
    label: "render pipeline",
    layout: "auto",
    vertex: {
        module: shader,
        entryPoint: "vs",
    } as GPUVertexState,
    fragment: {
        module: shader,
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

//////////// CREATE BINDGROUP ////////////

const bindGroup: GPUBindGroup = device.createBindGroup({
    label: "bindgroup",
    layout: pipeline.getBindGroupLayout(0),
    entries: [
        {
            binding: 0,
            resource: { buffer: uniformBuffer } as GPUBindingResource,
        } as GPUBindGroupEntry,
        {
            binding: 1,
            resource: { buffer: vertexBuffer } as GPUBindingResource,
        } as GPUBindGroupEntry,
        {
            binding: 2,
            resource: { buffer: instanceBuffer } as GPUBindingResource,
        } as GPUBindGroupEntry,
    ],
} as GPUBindGroupDescriptor);

//////////// CREATE BUNDLE ////////////

const bundleEncoder: GPURenderBundleEncoder = device.createRenderBundleEncoder({
    label: "render bundle",
    colorFormats: [presentationFormat],
    depthStencilFormat: "depth24plus",
} as GPURenderBundleEncoderDescriptor);
bundleEncoder.setPipeline(pipeline);
bundleEncoder.setBindGroup(0, bindGroup);
bundleEncoder.setIndexBuffer(indexBuffer, "uint32");
geometries.forEach((_geometry: OBJParseResult, i: int) => {
    bundleEncoder.drawIndexedIndirect(indirectBuffer, 20 * i);
});
const bundle: GPURenderBundle = bundleEncoder.finish();

//////////// SETUP RENDERPASS ////////////

const colorAttachment: GPURenderPassColorAttachment = {
    label: "color attachment",
    view: context!.getCurrentTexture().createView(),
    clearValue: [0.3, 0.3, 0.3, 1],
    loadOp: "clear",
    storeOp: "store",
} as GPURenderPassColorAttachment;

const depthTexture: GPUTexture = device.createTexture({
    label: "depth texture",
    size: [canvas.width, canvas.height],
    format: "depth24plus",
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
} as GPUTextureDescriptor);

const depthStencilAttachment: GPURenderPassDepthStencilAttachment = {
    label: "depth stencil attachment",
    view: depthTexture.createView(),
    depthClearValue: 1,
    depthLoadOp: "clear",
    depthStoreOp: "store",
} as GPURenderPassDepthStencilAttachment;

const renderPassDescriptor: GPURenderPassDescriptor = {
    label: "render pass",
    colorAttachments: [colorAttachment],
    depthStencilAttachment: depthStencilAttachment,
    timestampWrites: gpuTiming.timestampWrites,
} as GPURenderPassDescriptor;

//////////// EACH FRAME ////////////

async function frame(now: float): Promise<void> {
    stats.time("cpu delta");

    //////////// UPDATE CONTROL CAMERA UNIFORMS ////////////

    control.update();
    camera.update().store(uniformData, 0);
    device!.queue.writeBuffer(uniformBuffer, 0, uniformData);

    //////////// RENDER FRAME ////////////

    colorAttachment.view = context!.getCurrentTexture().createView();
    depthStencilAttachment.view = depthTexture.createView();

    const renderEncoder: GPUCommandEncoder = device!.createCommandEncoder({
        label: "render command encoder",
    } as GPUObjectDescriptorBase);

    const renderPass: GPURenderPassEncoder =
        renderEncoder.beginRenderPass(renderPassDescriptor);
    renderPass.executeBundles([bundle]);
    renderPass.end();

    gpuTiming.resolve(renderEncoder);

    const renderCommandBuffer: GPUCommandBuffer = renderEncoder.finish();
    device!.queue.submit([renderCommandBuffer]);

    //////////// UPDATE STATS ////////////

    stats.time("cpu delta", "cpu delta");
    cpuDelta.sample(stats.get("cpu delta")!);

    stats.set("frame delta", now - stats.get("frame delta")!);
    frameDelta.sample(stats.get("frame delta")!);

    gpuTiming.readback((ms: float) => {
        stats.set("gpu delta", ms);
        gpuDelta.sample(ms);
    });

    // prettier-ignore
    stats.update(`
        <b>frame rate: ${(1_000 / frameDelta.get()).toFixed(0)} fps</b><br>
        frame delta: ${frameDelta.get().toFixed(2)} ms<br>
        <br>
        <b>cpu rate: ${(1_000 / cpuDelta.get()).toFixed(0)} fps</b><br>
        cpu delta: ${cpuDelta.get().toFixed(2)} ms<br>
        <br>
        <b>gpu rate: ${(1_000 / gpuDelta.get()).toFixed(0)} fps</b><br>
        gpu delta: ${gpuDelta.get().toFixed(2)} ms<br>
    `);
    stats.set("frame delta", now);

    requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
