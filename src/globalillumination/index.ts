/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, December 2023
 */

import { int, float } from "../../types/utilities/utils.type.js";
import { Controller } from "../Controller.js";
import { Stats } from "../Stats.js";
import { RollingAverage } from "../RollingAverage.js";
import { createCanvas, loadOBJ, loadText } from "./helper.js";
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
const gpuTargetDelta: RollingAverage = new RollingAverage(60);
const gpuDeferredDelta: RollingAverage = new RollingAverage(60);

const targetGPUTiming: GPUTiming = new GPUTiming(device);
const deferredGPUTiming: GPUTiming = new GPUTiming(device);

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

//////////// SETUP UNIFORM ////////////

const uniformData: Float32Array = new Float32Array(4 * 4);
const uniformBuffer: GPUBuffer = device.createBuffer({
    label: "uniforms uniform buffer",
    size: uniformData.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
} as GPUBufferDescriptor);

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

//////////// SETUP INSTANCES ////////////

const n: int = 1;
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

//////////// LOAD SHADER ////////////

const targetShader: GPUShaderModule = device.createShaderModule({
    label: "target shader",
    code: await loadText("./shaders/globalillumination/targets.wgsl"),
} as GPUShaderModuleDescriptor);

const deferredShader: GPUShaderModule = device.createShaderModule({
    label: "deferred shader",
    code: await loadText("./shaders/globalillumination/deferred.wgsl"),
} as GPUShaderModuleDescriptor);

//////////// RENDER TARGETS ////////////

const colorFormat: GPUTextureFormat = "bgra8unorm";
const colorTarget: GPUTexture = device.createTexture({
    label: "color target texture",
    size: [canvas.width, canvas.height],
    format: colorFormat,
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
});

const depthFormat: GPUTextureFormat = "depth24plus";
const depthTarget: GPUTexture = device.createTexture({
    label: "depth target texture",
    size: [canvas.width, canvas.height],
    format: depthFormat,
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
} as GPUTextureDescriptor);

const normalFormat: GPUTextureFormat = "rgba16float";
const normalTarget: GPUTexture = device.createTexture({
    label: "normal target texture",
    size: [canvas.width, canvas.height],
    format: normalFormat,
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
});

const targetViews = {
    color: colorTarget.createView(),
    depth: depthTarget.createView(),
    normal: normalTarget.createView(),
};

//////////// CREATE PIPELINE ////////////

const targetPipeline: GPURenderPipeline =
    await device.createRenderPipelineAsync({
        label: "target pipeline",
        layout: "auto",
        vertex: {
            module: targetShader,
            entryPoint: "vs",
        } as GPUVertexState,
        fragment: {
            module: targetShader,
            entryPoint: "fs",
            targets: [{ format: colorFormat }, { format: normalFormat }],
        } as GPUFragmentState,
        primitive: {
            topology: "triangle-list",
            cullMode: "back",
        } as GPUPrimitiveState,
        depthStencil: {
            depthWriteEnabled: true,
            depthCompare: "less",
            format: depthFormat,
        } as GPUDepthStencilState,
    } as GPURenderPipelineDescriptor);

const deferredPipeline: GPURenderPipeline =
    await device.createRenderPipelineAsync({
        label: "deferred pipeline",
        layout: "auto",
        vertex: {
            module: deferredShader,
            entryPoint: "vs",
        } as GPUVertexState,
        fragment: {
            module: deferredShader,
            entryPoint: "fs",
            targets: [{ format: presentationFormat }],
        } as GPUFragmentState,
        primitive: {
            topology: "triangle-list",
            cullMode: "back",
        } as GPUPrimitiveState,
    } as GPURenderPipelineDescriptor);

//////////// CREATE BINDGROUP ////////////

const targetBindGroup: GPUBindGroup = device.createBindGroup({
    label: "target bindgroup",
    layout: targetPipeline.getBindGroupLayout(0),
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

const deferredBindGroup: GPUBindGroup = device.createBindGroup({
    label: "deferred bindgroup",
    layout: deferredPipeline.getBindGroupLayout(0),
    entries: [
        { binding: 0, resource: targetViews.color } as GPUBindGroupEntry,
        { binding: 1, resource: targetViews.depth } as GPUBindGroupEntry,
        { binding: 2, resource: targetViews.normal } as GPUBindGroupEntry,
    ],
} as GPUBindGroupDescriptor);

//////////// SETUP RENDERPASS DESCRIPTORS ////////////

const colorTargetAttachment: GPURenderPassColorAttachment = {
    label: "color target attachment",
    view: targetViews.color,
    clearValue: [0, 0, 0, 1],
    loadOp: "clear",
    storeOp: "store",
} as GPURenderPassColorAttachment;

const depthStencilAttachment: GPURenderPassDepthStencilAttachment = {
    label: "depth stencil attachment",
    view: targetViews.depth,
    depthClearValue: 1,
    depthLoadOp: "clear",
    depthStoreOp: "store",
} as GPURenderPassDepthStencilAttachment;

const normalTargetAttachment: GPURenderPassColorAttachment = {
    label: "normal target attachment",
    view: targetViews.normal,
    clearValue: [0, 0, 0, 1],
    loadOp: "clear",
    storeOp: "store",
} as GPURenderPassColorAttachment;

const targetPassDescriptor: GPURenderPassDescriptor = {
    label: "target render pass",
    colorAttachments: [colorTargetAttachment, normalTargetAttachment],
    depthStencilAttachment: depthStencilAttachment,
    timestampWrites: targetGPUTiming.timestampWrites,
} as GPURenderPassDescriptor;

const finalColorAttachment: GPURenderPassColorAttachment = {
    label: "final color attachment",
    view: context!.getCurrentTexture().createView(),
    clearValue: [0.3, 0.3, 0.3, 1],
    loadOp: "clear",
    storeOp: "store",
} as GPURenderPassColorAttachment;

const deferredPassDescriptor: GPURenderPassDescriptor = {
    label: "deferred render pass",
    colorAttachments: [finalColorAttachment],
    timestampWrites: deferredGPUTiming.timestampWrites,
} as GPURenderPassDescriptor;

//////////// CREATE BUNDLE ////////////

const targetBundleEncoder: GPURenderBundleEncoder =
    device.createRenderBundleEncoder({
        label: "target bundle",
        colorFormats: [colorFormat, normalFormat],
        depthStencilFormat: depthFormat,
    } as GPURenderBundleEncoderDescriptor);
targetBundleEncoder.setPipeline(targetPipeline);
targetBundleEncoder.setBindGroup(0, targetBindGroup);
targetBundleEncoder.setIndexBuffer(indexBuffer, "uint32");
geometries.forEach((_geometry: OBJParseResult, i: int) => {
    targetBundleEncoder.drawIndexedIndirect(indirectBuffer, 20 * i);
});
const targetBundle: GPURenderBundle = targetBundleEncoder.finish();

const deferredBundleEncoder: GPURenderBundleEncoder =
    device.createRenderBundleEncoder({
        label: "deferred bundle",
        colorFormats: [presentationFormat],
    } as GPURenderBundleEncoderDescriptor);
deferredBundleEncoder.setPipeline(deferredPipeline);
deferredBundleEncoder.setBindGroup(0, deferredBindGroup);
deferredBundleEncoder.draw(6);
const deferredBundle: GPURenderBundle = deferredBundleEncoder.finish();

//////////// EACH FRAME ////////////

async function frame(now: float): Promise<void> {
    stats.time("cpu delta");

    //////////// UPDATE CONTROL CAMERA UNIFORMS ////////////

    control.update();
    camera.update().store(uniformData, 0);
    device!.queue.writeBuffer(uniformBuffer, 0, uniformData);

    //////////// RENDER FRAME ////////////

    finalColorAttachment.view = context!.getCurrentTexture().createView();

    const renderEncoder: GPUCommandEncoder = device!.createCommandEncoder({
        label: "render command encoder",
    } as GPUObjectDescriptorBase);

    const targetPass: GPURenderPassEncoder =
        renderEncoder.beginRenderPass(targetPassDescriptor);
    targetPass.executeBundles([targetBundle]);
    targetPass.end();

    const deferredPass: GPURenderPassEncoder = renderEncoder.beginRenderPass(
        deferredPassDescriptor,
    );
    deferredPass.executeBundles([deferredBundle]);
    deferredPass.end();

    targetGPUTiming.resolve(renderEncoder);
    deferredGPUTiming.resolve(renderEncoder);

    const renderCommandBuffer: GPUCommandBuffer = renderEncoder.finish();
    device!.queue.submit([renderCommandBuffer]);

    //////////// UPDATE STATS ////////////

    stats.time("cpu delta", "cpu delta");
    cpuDelta.sample(stats.get("cpu delta")!);

    stats.set("frame delta", now - stats.get("frame delta")!);
    frameDelta.sample(stats.get("frame delta")!);

    targetGPUTiming.readback((ms: float) => {
        stats.set("gpu target delta", ms);
        gpuTargetDelta.sample(ms);
    });
    deferredGPUTiming.readback((ms: float) => {
        stats.set("gpu deferred delta", ms);
        gpuDeferredDelta.sample(ms);
    });
    const gpuDeltaSum: float = gpuTargetDelta.get() + gpuDeferredDelta.get();

    // prettier-ignore
    stats.update(`
        <b>frame rate: ${(1_000 / frameDelta.get()).toFixed(0)} fps</b><br>
        frame delta: ${frameDelta.get().toFixed(2)} ms<br>
        <br>
        <b>cpu rate: ${(1_000 / cpuDelta.get()).toFixed(0)} fps</b><br>
        cpu delta: ${cpuDelta.get().toFixed(2)} ms<br>
        <br>
        <b>gpu rate: ${(1_000 / gpuDeltaSum).toFixed(0)} fps</b><br>
        gpu delta: ${gpuDeltaSum.toFixed(2)} ms<br>
         - target delta: ${gpuTargetDelta.get().toFixed(2)} ms<br>
         - deferred delta: ${gpuDeferredDelta.get().toFixed(2)} ms<br>
    `);
    stats.set("frame delta", now);

    requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
