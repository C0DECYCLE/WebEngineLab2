/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, October 2023
 */

import { int, float } from "../../types/utilities/utils.type.js";
import { toRadian, dotit } from "../utilities/utils.js";
import { log } from "../utilities/logger.js";
import { Vec3 } from "../utilities/Vec3.js";
import { Mat4 } from "../utilities/Mat4.js";
import { Controller } from "../Controller.js";
//import { OBJParseResult, OBJParser } from "../OBJParser.js";
import { Stats } from "../Stats.js";
import { RollingAverage } from "../RollingAverage.js";
import {
    ChunkGeometryData,
    ChunkGeometry,
    ChunkInstanceLength,
} from "./chunk.js";
import { MaxTreeLength, TreeData, generateTree, storeTree } from "./tree.js";
import { byteSize, createGPU, loadShader } from "./helper.js";
import { createTerrainShader } from "./terrain.js";

//////////// SETUP ////////////

export const { canvas, device, context, presentationFormat } =
    await createGPU();

//////////// SHADER ////////////

const commonShader: GPUShaderModule = await loadShader(
    device,
    "./shaders/terrain/common.wgsl",
);

const terrainShader: GPUShaderModule = await createTerrainShader(device);

//////////// GPU TIMING ////////////

const capacity: int = 2;

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

//////////// TEXTURE ////////////

async function loadImageBitmap(url: string): Promise<ImageBitmap> {
    const res: Response = await fetch(url);
    const blob: Blob = await res.blob();
    return await createImageBitmap(blob, { colorSpaceConversion: "none" });
}

const url: string = "./resources/heightmap-self0.png";
const source: ImageBitmap = await loadImageBitmap(url);
const texture: GPUTexture = device.createTexture({
    label: "heightmap texture",
    format: "rgba8unorm",
    size: [source.width, source.height],
    usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.RENDER_ATTACHMENT,
} as GPUTextureDescriptor);

device.queue.copyExternalImageToTexture(
    {
        source: source,
    } as GPUImageCopyExternalImage,
    {
        texture: texture,
    } as GPUImageCopyTextureTagged,
    {
        width: source.width,
        height: source.height,
    } as GPUExtent3DStrict,
);

//////////// PIPELINE ////////////

const renderPipeline: GPURenderPipeline = device.createRenderPipeline({
    label: "render pipeline",
    layout: "auto",
    vertex: {
        module: terrainShader,
        entryPoint: "vs",
    } as GPUVertexState,
    fragment: {
        module: terrainShader,
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
    timestampWrites: {
        querySet: querySet,
        beginningOfPassWriteIndex: 0,
        endOfPassWriteIndex: 1,
    } as GPURenderPassTimestampWrites,
} as GPURenderPassDescriptor;

//////////// UNIFORM ////////////

const uniformFloats: int = 4 * 4;
const uniformData: Float32Array = new Float32Array(uniformFloats);

const uniformArrayBuffer: ArrayBuffer = uniformData.buffer;
const uniformBuffer: GPUBuffer = device.createBuffer({
    label: "uniforms uniform buffer",
    size: uniformArrayBuffer.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
} as GPUBufferDescriptor);

device!.queue.writeBuffer(uniformBuffer, 0, uniformArrayBuffer);

//////////// VERTECIES INDICES ////////////

/*
const raw: string = await fetch("./resources/cube.obj").then(
    async (response: Response) => await response.text(),
);
const parser: OBJParser = new OBJParser();
const data: OBJParseResult = parser.parse(raw, true);
*/
const data: ChunkGeometryData = ChunkGeometry;

const verteciesCount: int = data.positions.length / 4;
const indicesCount: int = data.indicies.length;

const vertexArrayBuffer: ArrayBuffer = data.positions.buffer;
const verteciesBuffer: GPUBuffer = device.createBuffer({
    label: "vertex buffer",
    size: vertexArrayBuffer.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
} as GPUBufferDescriptor);

device.queue.writeBuffer(verteciesBuffer, 0, vertexArrayBuffer);

const indexArrayBuffer: ArrayBuffer = data.indicies!.buffer;
const indicesBuffer: GPUBuffer = device.createBuffer({
    label: "index buffer",
    size: indexArrayBuffer.byteLength,
    usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
} as GPUBufferDescriptor);

device.queue.writeBuffer(indicesBuffer, 0, indexArrayBuffer);

log("vertecies", dotit(verteciesCount));
log("indices", dotit(indicesCount));

//////////// INSTANCES ////////////

const instancesData: Float32Array = new Float32Array(
    MaxTreeLength * ChunkInstanceLength,
);

const instancesArrayBuffer: ArrayBuffer = instancesData.buffer;
const instancesBuffer: GPUBuffer = device.createBuffer({
    label: "instances buffer",
    size: instancesArrayBuffer.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
} as GPUBufferDescriptor);

//log("instances", dotit(instancesCount));

//////////// INDIRECT ////////////

const indirectData: Uint32Array = new Uint32Array([
    indicesCount, //aka indexCount
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

const indirectReadbackBuffer: GPUBuffer = device.createBuffer({
    size: indirectBuffer.size,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
});

//////////// BINDGROUP ////////////

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
            resource: { buffer: instancesBuffer } as GPUBindingResource,
        } as GPUBindGroupEntry,
        {
            binding: 3,
            resource: texture.createView(),
        } as GPUBindGroupEntry,
    ],
} as GPUBindGroupDescriptor);

//////////// MATRIX ////////////

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

//////////// CONTROL ////////////

const control: Controller = new Controller(canvas, {
    position: cameraPos,
    direction: cameraDir,
});

//////////// STATS ////////////

const stats: Stats = new Stats();
stats.set("frame delta", 0);
stats.set("gpu delta", 0);
stats.set("render delta", 0);
stats.set("instances", 0);
stats.show();

const frameDelta: RollingAverage = new RollingAverage(60);
const cpuDelta: RollingAverage = new RollingAverage(60);
const gpuDelta: RollingAverage = new RollingAverage(60);
const renderDelta: RollingAverage = new RollingAverage(60);

//////////// RENDER BUNDLE ////////////

const renderBundleEncoder: GPURenderBundleEncoder =
    device.createRenderBundleEncoder({
        label: "render bundle",
        colorFormats: [presentationFormat],
        depthStencilFormat: "depth24plus",
    } as GPURenderBundleEncoderDescriptor);

renderBundleEncoder.setPipeline(renderPipeline);
renderBundleEncoder.setBindGroup(0, renderBindGroup);
renderBundleEncoder.setIndexBuffer(indicesBuffer, "uint32");
renderBundleEncoder.drawIndexedIndirect(indirectBuffer, 0);

const renderBundle: GPURenderBundle = renderBundleEncoder.finish();

async function render(now: float): Promise<void> {
    stats.time("cpu delta");

    //////////// UPDATE ////////////

    control.update();
    cameraView.view(cameraPos, cameraDir, up);
    viewProjection.multiply(cameraView, projection).store(uniformData, 0);
    device!.queue.writeBuffer(uniformBuffer, 0, uniformArrayBuffer);

    const tree: TreeData = generateTree(new Vec3(), 16, cameraPos);
    const instancesCount: int = storeTree(instancesData, tree);
    const len: int = instancesCount * ChunkInstanceLength * byteSize;
    device!.queue.writeBuffer(instancesBuffer, 0, instancesArrayBuffer, 0, len);
    indirectData[1] = instancesCount;
    device!.queue.writeBuffer(indirectBuffer, 0, indirectArrayBuffer);

    //////////// RENDER ////////////

    colorAttachment.view = context!.getCurrentTexture().createView();
    depthStencilAttachment.view = depthTexture.createView();

    device!.queue.writeBuffer(indirectBuffer, 0, indirectArrayBuffer);

    const renderEncoder: GPUCommandEncoder = device!.createCommandEncoder({
        label: "render command encoder",
    } as GPUObjectDescriptorBase);

    const renderPass: GPURenderPassEncoder =
        renderEncoder.beginRenderPass(renderPassDescriptor);
    renderPass.executeBundles([renderBundle]);
    renderPass.end();

    if (indirectReadbackBuffer.mapState === "unmapped") {
        renderEncoder.copyBufferToBuffer(
            indirectBuffer,
            0,
            indirectReadbackBuffer,
            0,
            indirectReadbackBuffer.size,
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

    if (indirectReadbackBuffer.mapState === "unmapped") {
        indirectReadbackBuffer.mapAsync(GPUMapMode.READ).then(() => {
            const result: Uint32Array = new Uint32Array(
                indirectReadbackBuffer.getMappedRange().slice(0),
            );
            indirectReadbackBuffer.unmap();
            stats.set("instances", result[1]);
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
                Number(timingsNanoseconds[1] - timingsNanoseconds[0]) /
                    1_000_000,
            );
            gpuDelta.sample(stats.get("gpu delta")!);
            stats.set(
                "render delta",
                Number(timingsNanoseconds[1] - timingsNanoseconds[0]) /
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
        |- render delta: ${renderDelta.get().toFixed(2)} ms<br>
        <br>
        draw calls: 1<br>
        instances: ${stats.get("instances")}<br>
        vertices: ${verteciesCount * stats.get("instances")!}<br>
        indices: ${indicesCount * stats.get("instances")!}
    `);
    stats.set("frame delta", now);

    requestAnimationFrame(render);
}
requestAnimationFrame(render);