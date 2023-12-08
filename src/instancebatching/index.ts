/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, October 2023
 */

import { int, float } from "../../types/utilities/utils.type.js";
import { Controller } from "../Controller.js";
import { Stats } from "../Stats.js";
import { RollingAverage } from "../RollingAverage.js";
import { createCanvas } from "./helper.js";
import { Camera } from "./Camera.js";
import { GPUTiming } from "./GPUTiming.js";
import { Geometry } from "./Geometry.js";
//import { log } from "../utilities/logger.js";

//////////// SETUP GPU ////////////

const canvas: HTMLCanvasElement = createCanvas();
const adapter: GPUAdapter = (await navigator.gpu?.requestAdapter())!;
const device: GPUDevice = (await adapter?.requestDevice({
    requiredFeatures: ["timestamp-query"],
} as GPUDeviceDescriptor))!;
const context: GPUCanvasContext = canvas.getContext("webgpu")!;
const presentationFormat: GPUTextureFormat =
    navigator.gpu.getPreferredCanvasFormat();
context.configure({
    device: device,
    format: presentationFormat,
} as GPUCanvasConfiguration);

//////////// SETUP CAMERA CONTROL ////////////

const camera: Camera = new Camera(canvas.width / canvas.height, 1000);
const control: Controller = new Controller(canvas, camera);

//////////// SETUP STATS ////////////

const stats: Stats = new Stats();
stats.set("frame delta", 0);
stats.set("gpu delta", 0);
stats.show();

const frameDelta: RollingAverage = new RollingAverage(60);
const cpuDelta: RollingAverage = new RollingAverage(60);
const gpuDelta: RollingAverage = new RollingAverage(60);

const gpuTiming: GPUTiming = new GPUTiming(device);

//////////// SETUP UNIFORMS ////////////

const uniformFloats: int = 4 * 4;
const uniformData: Float32Array = new Float32Array(uniformFloats);

const uniformBuffer: GPUBuffer = device.createBuffer({
    label: "uniforms uniform buffer",
    size: uniformData.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
} as GPUBufferDescriptor);

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

//////////// SETUP SHADER ////////////

const shader: GPUShaderModule = device.createShaderModule({
    label: "shader",
    code: await fetch("./shaders/instancebatching/old.wgsl").then(
        async (response: Response) => await response.text(),
    ),
} as GPUShaderModuleDescriptor);

//////////// SETUP PIPELINE ////////////

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

//////////// SETUP GEOMETRY ////////////

const n: int = 10_000;

const geos: Geometry[] = [];

const cube: Geometry = new Geometry(
    device,
    "cube.obj",
    n,
    uniformBuffer,
    pipeline,
);
geos.push(cube);

const ico: Geometry = new Geometry(
    device,
    "icosphere.obj",
    n,
    uniformBuffer,
    pipeline,
);
geos.push(ico);

const torus: Geometry = new Geometry(
    device,
    "torus.obj",
    n,
    uniformBuffer,
    pipeline,
);
geos.push(torus);

const cylinder: Geometry = new Geometry(
    device,
    "cylinder.obj",
    n,
    uniformBuffer,
    pipeline,
);
geos.push(cylinder);

const cone: Geometry = new Geometry(
    device,
    "cone.obj",
    n,
    uniformBuffer,
    pipeline,
);
geos.push(cone);

const suzanne: Geometry = new Geometry(
    device,
    "suzanne.obj",
    n,
    uniformBuffer,
    pipeline,
);
geos.push(suzanne);

//////////// EACH FRAME ////////////

async function frame(now: float): Promise<void> {
    stats.time("cpu delta");

    //////////// UPDATE CONTROL CAMERA UNIFORMS ////////////

    control.update();
    camera.update().store(uniformData, 0);
    device!.queue.writeBuffer(uniformBuffer, 0, uniformData.buffer);

    //////////// RENDER FRAME ////////////

    colorAttachment.view = context!.getCurrentTexture().createView();
    depthStencilAttachment.view = depthTexture.createView();

    const renderEncoder: GPUCommandEncoder = device!.createCommandEncoder({
        label: "render command encoder",
    } as GPUObjectDescriptorBase);

    const renderPass: GPURenderPassEncoder =
        renderEncoder.beginRenderPass(renderPassDescriptor);
    renderPass.executeBundles(
        geos.flatMap((geo: Geometry) => (geo.bundle ? [geo.bundle] : [])),
    );
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