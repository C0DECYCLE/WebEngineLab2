/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, October 2023
 */

import {
    int,
    float,
    Nullable,
    Undefinable,
} from "../../types/utilities/utils.type.js";
import { toRadian, dotit } from "../utilities/utils.js";
import { log } from "../utilities/logger.js";
import { Vec3 } from "../utilities/Vec3.js";
import { Mat4 } from "../utilities/Mat4.js";
import { Controller } from "../Controller.js";
import { OBJParseResult, OBJParser } from "../OBJParser.js";
import { Stats } from "../Stats.js";
import { RollingAverage } from "../RollingAverage.js";

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
} as GPUDeviceDescriptor);
const context: Nullable<GPUCanvasContext> = canvas.getContext("webgpu");
if (!device || !context) {
    throw new Error("Browser doesn't support WebGPU.");
}
const presentationFormat: GPUTextureFormat =
    navigator.gpu.getPreferredCanvasFormat();
context.configure({
    device: device,
    format: presentationFormat,
} as GPUCanvasConfiguration);

//const sampleCount: int = 4;

//////////// SHADER ////////////

const computeModule: GPUShaderModule = device.createShaderModule({
    label: "compute shader",
    code: await fetch("./shaders/grass/compute.wgsl").then(
        async (response: Response) => await response.text(),
    ),
} as GPUShaderModuleDescriptor);

const cullModule: GPUShaderModule = device.createShaderModule({
    label: "cull shader",
    code: await fetch("./shaders/grass/cull.wgsl").then(
        async (response: Response) => await response.text(),
    ),
} as GPUShaderModuleDescriptor);

const renderModule: GPUShaderModule = device.createShaderModule({
    label: "render shader",
    code: await fetch("./shaders/grass/render.wgsl").then(
        async (response: Response) => await response.text(),
    ),
} as GPUShaderModuleDescriptor);

//////////// CONSTS ////////////

const byteSize: int = 4;

//////////// GPU TIMING ////////////

const capacity: int = 4;

const querySet: GPUQuerySet = device.createQuerySet({
    type: "timestamp",
    count: capacity,
} as GPUQuerySetDescriptor);

const queryBuffer: GPUBuffer = device.createBuffer({
    size: capacity * (byteSize * 2), //64bit
    usage:
        GPUBufferUsage.QUERY_RESOLVE |
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST,
} as GPUBufferDescriptor);

const queryReadbackBuffer: GPUBuffer = device.createBuffer({
    size: queryBuffer.size,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
});

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
    /*
    multisample: {
        count: sampleCount,
    } as GPUMultisampleState,
    */
} as GPURenderPipelineDescriptor);

//////////// RENDERPASS ////////////
/*
const viewTexture: GPUTexture = device.createTexture({
    label: "view texture",
    size: [canvas.width, canvas.height],
    sampleCount: sampleCount,
    format: presentationFormat,
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
} as GPUTextureDescriptor);
*/

const colorAttachment: GPURenderPassColorAttachment = {
    label: "color attachment",
    view: context!.getCurrentTexture().createView(), //viewTexture.createView(),
    //resolveTarget: context!.getCurrentTexture().createView(),
    clearValue: [0.3, 0.3, 0.3, 1],
    loadOp: "clear",
    storeOp: "store", //"discard",
} as GPURenderPassColorAttachment;

const depthTexture: GPUTexture = device.createTexture({
    label: "depth texture",
    size: [canvas.width, canvas.height],
    //sampleCount: sampleCount,
    format: "depth24plus",
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
} as GPUTextureDescriptor);

const depthStencilAttachment: GPURenderPassDepthStencilAttachment = {
    label: "depth stencil attachment",
    view: depthTexture.createView(),
    depthClearValue: 1,
    depthLoadOp: "clear",
    depthStoreOp: "store", //"discard",
} as GPURenderPassDepthStencilAttachment;

const renderPassDescriptor: GPURenderPassDescriptor = {
    label: "render pass",
    colorAttachments: [colorAttachment],
    depthStencilAttachment: depthStencilAttachment,
    timestampWrites: {
        querySet: querySet,
        beginningOfPassWriteIndex: 2,
        endOfPassWriteIndex: 3,
    } as GPURenderPassTimestampWrites,
} as GPURenderPassDescriptor;

//////////// UNIFORM ////////////

const uniformBufferSize: int = (4 * 4 + (1 + 3)) * byteSize;

/*(4 * byteSize - (uniformBufferSize % (4 * byteSize)))*/
const uniformBuffer: GPUBuffer = device.createBuffer({
    label: "uniforms uniform buffer",
    size: uniformBufferSize,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
} as GPUBufferDescriptor);

const uniformArrayBuffer: ArrayBuffer = new ArrayBuffer(uniformBufferSize);
const floatValues: Float32Array = new Float32Array(uniformArrayBuffer);
//const intValues: Uint32Array = new Uint32Array(uniformArrayBuffer);

const matrixOffset: int = 0;
const timeOffset: int = 16;
//const modeOffset: int = 17;

/*
(window as any).setMode = (mode: 0 | 1 | 2) => {
    intValues[modeOffset] = mode;
    device!.queue.writeBuffer(
        uniformBuffer,
        modeOffset * byteSize,
        uniformArrayBuffer,
        modeOffset * byteSize,
    );
};
(window as any).setMode(2);
*/

//////////// VERTICES ////////////

const raw: string = await fetch("./resources/grass.obj").then(
    async (response: Response) => await response.text(),
);
const parser: OBJParser = new OBJParser();
const data: OBJParseResult = parser.parse(raw, true);
const vertexCount: int = data.indices!.length;

//log(parser.parse(raw));
//log(parser.parse(raw, true));

const vertexArrayBuffer: ArrayBuffer = data.vertices.buffer;
const verticesBuffer: GPUBuffer = device.createBuffer({
    label: "vertices storage buffer",
    size: vertexArrayBuffer.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
} as GPUBufferDescriptor);

const indexArrayBuffer: ArrayBuffer = data.indices!.buffer;
const indicesBuffer: GPUBuffer = device.createBuffer({
    label: "index buffer",
    size: indexArrayBuffer.byteLength,
    usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
} as GPUBufferDescriptor);

device.queue.writeBuffer(indicesBuffer, 0, indexArrayBuffer);
device.queue.writeBuffer(verticesBuffer, 0, vertexArrayBuffer);

log(dotit(vertexCount));

//////////// INSTANCES ////////////

const instanceCount: int = 10_000;
const instanceByteLength: int = ((3 + 1) * 3 + (3 + 1)) * byteSize;

const instancesBuffer: GPUBuffer = device.createBuffer({
    label: "instances storage buffer",
    size: instanceCount * instanceByteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
} as GPUBufferDescriptor);

const cullBuffer: GPUBuffer = device.createBuffer({
    label: "cull storage buffer",
    size: instanceCount * instanceByteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
} as GPUBufferDescriptor);

log(dotit(instanceCount));

//////////// INDIRECT ////////////

const indirectData: Uint32Array = new Uint32Array([
    vertexCount, //aka indexCount
    0, //instanceCount,
    0,
    0,
    0,
]);

const indirectArrayBuffer: ArrayBuffer = indirectData.buffer;
const indirectBuffer: GPUBuffer = device.createBuffer({
    label: "indirect buffer",
    size: 5 * byteSize,
    usage:
        GPUBufferUsage.INDIRECT |
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST,
} as GPUBufferDescriptor);

device.queue.writeBuffer(indirectBuffer, 0, indirectArrayBuffer);

const readbackBuffer: GPUBuffer = device.createBuffer({
    size: indirectBuffer.size,
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
            resource: { buffer: verticesBuffer } as GPUBindingResource,
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
device!.queue.submit([computeCommandBuffer]);

//////////// MATRIX ////////////

const cameraView: Mat4 = new Mat4();
const projection: Mat4 = Mat4.Perspective(
    60 * toRadian,
    canvas.width / canvas.height,
    0.01,
    1000,
);
const viewProjection: Mat4 = new Mat4();

const cameraPos: Vec3 = new Vec3(0, 6, 2);
const cameraDir: Vec3 = new Vec3(0, 0.5, 1).normalize();
const up: Vec3 = new Vec3(0, 1, 0);

//////////// CONTROL ////////////

const control: Controller = new Controller(canvas, {
    position: cameraPos,
    direction: cameraDir,
});

//////////// STATS ////////////

const stats: Stats = new Stats();
stats.set("frame delta", 0);
stats.set("gpu delta", 0);
stats.set("cull delta", 0);
stats.set("render delta", 0);
stats.set("cull", 0);
stats.show();

const frameDelta: RollingAverage = new RollingAverage(60);
const cpuDelta: RollingAverage = new RollingAverage(60);
const gpuDelta: RollingAverage = new RollingAverage(60);
const cullDelta: RollingAverage = new RollingAverage(60);
const renderDelta: RollingAverage = new RollingAverage(60);

//////////// RENDER BUNDLE ////////////

function draw(
    renderEncoder: GPURenderPassEncoder | GPURenderBundleEncoder,
): void {
    renderEncoder.setPipeline(renderPipeline);
    renderEncoder.setBindGroup(0, renderBindGroup);
    renderEncoder.setIndexBuffer(indicesBuffer, "uint32");
    renderEncoder.drawIndexedIndirect(indirectBuffer, 0);
    /*
    for (let i: int = 0; i < instanceCount; i++) {
        //out of date! now indexed!
        renderEncoder.draw(vertexCount, 1, 0, i);
    }
    */
}

const renderBundleEncoder: GPURenderBundleEncoder =
    device.createRenderBundleEncoder({
        label: "render bundle",
        colorFormats: [presentationFormat],
        //sampleCount: sampleCount,
        depthStencilFormat: "depth24plus",
    } as GPURenderBundleEncoderDescriptor);
draw(renderBundleEncoder);
const renderBundle: GPURenderBundle = renderBundleEncoder.finish();

async function render(now: float): Promise<void> {
    stats.time("cpu delta");

    //////////// UPDATE ////////////

    control.update();

    cameraView.view(cameraPos, cameraDir, up);
    viewProjection
        .multiply(cameraView, projection)
        .store(floatValues, matrixOffset);

    floatValues[timeOffset] = performance.now();

    device!.queue.writeBuffer(uniformBuffer, 0, uniformArrayBuffer);

    //////////// CULL DRAW ////////////

    colorAttachment.view = context!.getCurrentTexture().createView();
    //colorAttachment.resolveTarget = context!.getCurrentTexture().createView();
    depthStencilAttachment.view = depthTexture.createView();

    device!.queue.writeBuffer(indirectBuffer, 0, indirectArrayBuffer);

    const renderEncoder: GPUCommandEncoder = device!.createCommandEncoder({
        label: "render command encoder",
    } as GPUObjectDescriptorBase);

    const cullPass: GPUComputePassEncoder = renderEncoder.beginComputePass({
        label: "cull pass",
        timestampWrites: {
            querySet: querySet,
            beginningOfPassWriteIndex: 0,
            endOfPassWriteIndex: 1,
        } as GPURenderPassTimestampWrites,
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

    if (readbackBuffer.mapState === "unmapped") {
        renderEncoder.copyBufferToBuffer(
            indirectBuffer,
            0,
            readbackBuffer,
            0,
            readbackBuffer.size,
        );
    }

    renderEncoder.resolveQuerySet(querySet, 0, capacity, queryBuffer, 0);

    if (queryReadbackBuffer.mapState === "unmapped") {
        renderEncoder.copyBufferToBuffer(
            queryBuffer,
            0,
            queryReadbackBuffer,
            0,
            queryReadbackBuffer.size,
        );
    }

    const renderCommandBuffer: GPUCommandBuffer = renderEncoder.finish();
    device!.queue.submit([renderCommandBuffer]);

    if (readbackBuffer.mapState === "unmapped") {
        readbackBuffer.mapAsync(GPUMapMode.READ).then(() => {
            const result: Uint32Array = new Uint32Array(
                readbackBuffer.getMappedRange().slice(0),
            );
            readbackBuffer.unmap();
            stats.set("cull", result[1]);
        });
    }

    if (queryReadbackBuffer.mapState === "unmapped") {
        queryReadbackBuffer.mapAsync(GPUMapMode.READ).then(() => {
            const timingsNanoseconds: BigInt64Array = new BigInt64Array(
                queryReadbackBuffer.getMappedRange().slice(0),
            );
            queryReadbackBuffer.unmap();
            stats.set(
                "gpu delta",
                Number(timingsNanoseconds[3] - timingsNanoseconds[0]) /
                    1_000_000,
            );
            gpuDelta.sample(stats.get("gpu delta")!);
            stats.set(
                "cull delta",
                Number(timingsNanoseconds[1] - timingsNanoseconds[0]) /
                    1_000_000,
            );
            cullDelta.sample(stats.get("cull delta")!);
            stats.set(
                "render delta",
                Number(timingsNanoseconds[3] - timingsNanoseconds[2]) /
                    1_000_000,
            );
            renderDelta.sample(stats.get("render delta")!);
        });
    }

    //////////// FRAME ////////////

    stats.set("frame delta", now - stats.get("frame delta")!);
    frameDelta.sample(stats.get("frame delta")!);
    stats.time("cpu delta", "cpu delta");
    cpuDelta.sample(stats.get("cpu delta")!);
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
        |- cull delta: ${cullDelta.get().toFixed(2)} ms<br>
        |- render delta: ${renderDelta.get().toFixed(2)} ms<br>
        <br>
        unculled: ${stats.get("cull")}
    `);
    stats.set("frame delta", now);

    requestAnimationFrame(render);
}
requestAnimationFrame(render);
