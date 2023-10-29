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
import { toRadian, dotit } from "./utilities/utils.js";
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

const computeModule: GPUShaderModule = device.createShaderModule({
    label: "compute shader",
    code: await fetch("./shaders/compute.wgsl").then(
        async (response: Response) => await response.text(),
    ),
} as GPUShaderModuleDescriptor);

const cullModule: GPUShaderModule = device.createShaderModule({
    label: "cull shader",
    code: await fetch("./shaders/cull.wgsl").then(
        async (response: Response) => await response.text(),
    ),
} as GPUShaderModuleDescriptor);

const renderModule: GPUShaderModule = device.createShaderModule({
    label: "render shader",
    code: await fetch("./shaders/render.wgsl").then(
        async (response: Response) => await response.text(),
    ),
} as GPUShaderModuleDescriptor);

//////////// PIPELINE ////////////

const computePipeline: GPUComputePipeline = device.createComputePipeline({
    label: "compute pipeline",
    layout: "auto",
    compute: {
        module: computeModule,
        entryPoint: "computeInstance",
    } as GPUProgrammableStage,
} as GPUComputePipelineDescriptor);

const cullPipeline: GPUComputePipeline = device.createComputePipeline({
    label: "cull pipeline",
    layout: "auto",
    compute: {
        module: cullModule,
        entryPoint: "cullInstance",
    } as GPUProgrammableStage,
} as GPUComputePipelineDescriptor);

const renderPipeline: GPURenderPipeline = device.createRenderPipeline({
    label: "render pipeline",
    layout: "auto",
    vertex: {
        module: renderModule,
        entryPoint: "vs",
    } as GPUVertexState,
    fragment: {
        module: renderModule,
        entryPoint: "fs",
        targets: [{ format: presentationFormat }],
    } as GPUFragmentState,
    primitive: {
        //cullMode: "back",
    } as GPUPrimitiveState,
    depthStencil: {
        depthWriteEnabled: true,
        depthCompare: "less",
        format: "depth24plus",
    } as GPUDepthStencilState,
} as GPURenderPipelineDescriptor);

//////////// RENDERPASS ////////////

const colorAttachment: GPURenderPassColorAttachment = {
    label: "color attachment",
    view: context!.getCurrentTexture().createView(),
    clearValue: [0.3, 0.3, 0.3, 1.0],
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
    depthClearValue: 1.0,
    depthLoadOp: "clear",
    depthStoreOp: "store",
} as GPURenderPassDepthStencilAttachment;

const renderPassDescriptor: GPURenderPassDescriptor = {
    label: "render pass",
    colorAttachments: [colorAttachment],
    depthStencilAttachment: depthStencilAttachment,
} as GPURenderPassDescriptor;

//////////// UNIFORM ////////////

const byteSize: int = 4;
const uniformBufferSize: int = 16 * byteSize + 1 * byteSize + 1 * byteSize;

const uniformBuffer: GPUBuffer = device.createBuffer({
    label: "uniforms uniform buffer",
    size:
        uniformBufferSize +
        (4 * byteSize - (uniformBufferSize % (4 * byteSize))),
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
} as GPUBufferDescriptor);

const uniformArrayBuffer: ArrayBuffer = new ArrayBuffer(uniformBufferSize);
const floatValues: Float32Array = new Float32Array(uniformArrayBuffer);
const intValues: Uint32Array = new Uint32Array(uniformArrayBuffer);

const matrixOffset: int = 0;
const modeOffset: int = 16;
const timeOffset: int = 17;

(window as any).setMode = (mode: 0 | 1 | 2) => {
    intValues[modeOffset] = mode;
    device?.queue.writeBuffer(
        uniformBuffer,
        modeOffset * byteSize,
        uniformArrayBuffer,
        modeOffset * byteSize,
    );
};
(window as any).setMode(2);

//////////// VERTECIES ////////////

const raw: string = await fetch("./resources/grass.obj").then(
    async (response: Response) => await response.text(),
);
const parser: OBJParser = new OBJParser();
const vertexData: Float32Array = parser.parse(raw);
const vertexCount: int = vertexData.length / 4;

const vertexArrayBuffer: ArrayBuffer = vertexData.buffer;
const verteciesBuffer: GPUBuffer = device.createBuffer({
    label: "vertices storage buffer",
    size: vertexArrayBuffer.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
} as GPUBufferDescriptor);

log(dotit(vertexCount));

device.queue.writeBuffer(verteciesBuffer, 0, vertexArrayBuffer);

//////////// INSTANCES ////////////

const instanceCount: int = 10_000;

const instancesBuffer: GPUBuffer = device.createBuffer({
    label: "instances storage buffer",
    size: instanceCount * 4 * 4 * byteSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
} as GPUBufferDescriptor);

const cullBuffer: GPUBuffer = device.createBuffer({
    label: "cull storage buffer",
    size: instanceCount * 4 * 4 * byteSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
} as GPUBufferDescriptor);

log(dotit(instanceCount));

//////////// INDIRECT ////////////

const indirectData: Uint32Array = new Uint32Array([
    vertexCount,
    0, //instanceCount,
    0,
    0,
]);

const indirectArrayBuffer: ArrayBuffer = indirectData.buffer;
const indirectBuffer: GPUBuffer = device.createBuffer({
    label: "indirect buffer",
    size: 4 * byteSize,
    usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.INDIRECT |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
} as GPUBufferDescriptor);

device.queue.writeBuffer(indirectBuffer, 0, indirectArrayBuffer);

const readbackBuffer: GPUBuffer = device.createBuffer({
    size: 4 * byteSize,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
});

//////////// BINDGROUP ////////////

const computeBindGroup: GPUBindGroup = device.createBindGroup({
    label: "compute bind group",
    layout: computePipeline.getBindGroupLayout(0),
    entries: [
        {
            binding: 0,
            resource: { buffer: instancesBuffer } as GPUBindingResource,
        } as GPUBindGroupEntry,
    ],
} as GPUBindGroupDescriptor);

const cullBindGroup: GPUBindGroup = device.createBindGroup({
    label: "cull bind group",
    layout: cullPipeline.getBindGroupLayout(0),
    entries: [
        {
            binding: 0,
            resource: { buffer: instancesBuffer } as GPUBindingResource,
        } as GPUBindGroupEntry,
        {
            binding: 1,
            resource: { buffer: indirectBuffer } as GPUBindingResource,
        } as GPUBindGroupEntry,
        {
            binding: 2,
            resource: { buffer: cullBuffer } as GPUBindingResource,
        } as GPUBindGroupEntry,
        {
            binding: 3,
            resource: { buffer: uniformBuffer } as GPUBindingResource,
        } as GPUBindGroupEntry,
    ],
} as GPUBindGroupDescriptor);

const renderBindGroup: GPUBindGroup = device.createBindGroup({
    label: "render bind group",
    layout: renderPipeline.getBindGroupLayout(0),
    entries: [
        {
            binding: 0,
            resource: { buffer: uniformBuffer } as GPUBindingResource,
        } as GPUBindGroupEntry,
        {
            binding: 1,
            resource: { buffer: verteciesBuffer } as GPUBindingResource,
        } as GPUBindGroupEntry,
        {
            binding: 2,
            resource: { buffer: cullBuffer } as GPUBindingResource,
        } as GPUBindGroupEntry,
    ],
} as GPUBindGroupDescriptor);

//////////// COMPUTE ////////////

const computeEncoder: GPUCommandEncoder = device!.createCommandEncoder({
    label: "compute command encoder",
} as GPUObjectDescriptorBase);

const computePass: GPUComputePassEncoder = computeEncoder.beginComputePass({
    label: "compute pass",
} as GPUComputePassDescriptor);
computePass.setPipeline(computePipeline);
computePass.setBindGroup(0, computeBindGroup);
computePass.dispatchWorkgroups(instanceCount / 100, 1, 1);
computePass.end();

const computeCommandBuffer: GPUCommandBuffer = computeEncoder.finish();
device?.queue.submit([computeCommandBuffer]);

//////////// MATRIX ////////////

const view: Mat4 = new Mat4();
const projection: Mat4 = Mat4.Perspective(
    60 * toRadian,
    canvas.width / canvas.height,
    0.01,
    1000.0,
);
const viewProjection: Mat4 = new Mat4();

const cameraPos: Vec3 = new Vec3(0.0, 6.0, 2.0);
const cameraDir: Vec3 = new Vec3(0.0, 0.5, 1.0).normalize();
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
stats.set("cull", 0);
stats.show();

//////////// RENDER BUNDLE ////////////

function draw(
    renderEncoder: GPURenderPassEncoder | GPURenderBundleEncoder,
): void {
    renderEncoder.setPipeline(renderPipeline);
    renderEncoder.setBindGroup(0, renderBindGroup);
    renderEncoder.drawIndirect(indirectBuffer, 0);
    /*
    for (let i: int = 0; i < instanceCount; i++) {
        renderEncoder.draw(vertexCount, 1, 0, i);
    }
    */
}

const renderBundleEncoder: GPURenderBundleEncoder =
    device.createRenderBundleEncoder({
        label: "render bundle",
        colorFormats: [presentationFormat],
        depthStencilFormat: "depth24plus",
    } as GPURenderBundleEncoderDescriptor);
draw(renderBundleEncoder);
const renderBundle: GPURenderBundle = renderBundleEncoder.finish();

async function render(now: float): Promise<void> {
    stats.time("cpu delta");

    //////////// UPDATE ////////////

    control.update();

    view.view(cameraPos, cameraDir, up);
    viewProjection.multiply(view, projection).store(floatValues, matrixOffset);
    device?.queue.writeBuffer(
        uniformBuffer,
        matrixOffset * byteSize,
        uniformArrayBuffer,
        matrixOffset * byteSize,
    );
    floatValues[timeOffset] = performance.now();
    device?.queue.writeBuffer(
        uniformBuffer,
        timeOffset * byteSize,
        uniformArrayBuffer,
        timeOffset * byteSize,
    );

    //////////// DRAW ////////////

    colorAttachment.view = context!.getCurrentTexture().createView();
    depthStencilAttachment.view = depthTexture.createView();

    const renderEncoder: GPUCommandEncoder = device!.createCommandEncoder({
        label: "render command encoder",
    } as GPUObjectDescriptorBase);

    const cullPass: GPUComputePassEncoder = renderEncoder.beginComputePass({
        label: "cull pass",
    } as GPUComputePassDescriptor);
    cullPass.setPipeline(cullPipeline);
    cullPass.setBindGroup(0, cullBindGroup);
    cullPass.dispatchWorkgroups(instanceCount / 100, 1, 1);
    cullPass.end();

    const renderPass: GPURenderPassEncoder =
        renderEncoder.beginRenderPass(renderPassDescriptor);
    const useRenderBundles: boolean = true;
    if (useRenderBundles) {
        renderPass.executeBundles([renderBundle]);
    } else {
        draw(renderPass);
    }
    renderPass.end();

    renderEncoder.copyBufferToBuffer(
        indirectBuffer,
        0,
        readbackBuffer,
        0,
        readbackBuffer.size,
    );

    const renderCommandBuffer: GPUCommandBuffer = renderEncoder.finish();
    device?.queue.submit([renderCommandBuffer]);

    await readbackBuffer.mapAsync(GPUMapMode.READ).then(() => {
        const result: Uint32Array = new Uint32Array(
            readbackBuffer.getMappedRange().slice(0),
        );
        readbackBuffer.unmap();
        //log(dotit(result[1]));
        stats.set("cull", result[1]);
    });

    //////////// FRAME ////////////

    stats.set("frame delta", now - stats.get("frame delta")!);
    stats.time("cpu delta", "cpu delta");
    // prettier-ignore
    stats.update(`
            <b>frame rate: ${(1_000 / stats.get("frame delta")!).toFixed(
                1,
            )} fps</b><br>
            frame delta: ${stats.get("frame delta")!.toFixed(2)} ms<br>
            <br>
            <b>cpu rate: ${(1_000 / stats.get("cpu delta")!).toFixed(
                1,
            )} fps</b><br>
            cpu delta: ${stats.get("cpu delta")!.toFixed(2)} ms<br><br>
            cull: ${stats.get("cull")}
    `);
    stats.set("frame delta", now);

    requestAnimationFrame(render);
}
requestAnimationFrame(render);
