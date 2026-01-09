/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, January 2025
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
import { OBJParseResult } from "../OBJParser.js";
import { Stats } from "../Stats.js";
import { RollingAverage } from "../RollingAverage.js";
/*
import {
    MeshoptClusterizer,
    // @ts-ignore
} from "../../../node_modules/meshoptimizer/meshopt_clusterizer.module.js";
 */
import { loadOBJ } from "../instancebatching/helper.js";
import { clusterizeTriangles, TriangleClusteringResult } from "./clusterize.js";
import { groupClusters } from "./group.js";

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
    requiredFeatures: ["timestamp-query", "primitive-index"],
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

//////////// SHADER ////////////

const computeShader: GPUShaderModule = device.createShaderModule({
    label: "compute shader",
    code: await fetch("./shaders/meshoptimizer/compute.wgsl").then(
        async (response: Response) => await response.text(),
    ),
} as GPUShaderModuleDescriptor);
const renderShader: GPUShaderModule = device.createShaderModule({
    label: "render shader",
    code: await fetch("./shaders/meshoptimizer/render.wgsl").then(
        async (response: Response) => await response.text(),
    ),
} as GPUShaderModuleDescriptor);

//////////// CONSTS ////////////

const byteSize: int = 4;
//const maxVertices: int = 255;
const maxTriangles: int = 128;
const numVertices: int = 3;
const vertexStride: int = 3 + 1;
const instanceStride: int = 3 + 1 + 1 + 3;
const instanceCount: int = 1; //125;
const spawnSize: int = 1; //5;
const maxTasks: int = 1_000_000;
const taskStride: int = 1 + 1;

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
        module: computeShader,
        entryPoint: "cs",
    } as GPUProgrammableStage,
} as GPUComputePipelineDescriptor);
const renderPipeline: GPURenderPipeline = device.createRenderPipeline({
    label: "render pipeline",
    layout: "auto",
    vertex: {
        module: renderShader,
        entryPoint: "vs",
    } as GPUVertexState,
    fragment: {
        module: renderShader,
        entryPoint: "fs",
        targets: [{ format: presentationFormat }],
    } as GPUFragmentState,
    primitive: {
        cullMode: "back",
    } as GPUPrimitiveState,
    depthStencil: {
        depthWriteEnabled: true,
        depthCompare: "less",
        format: "depth32float",
    } as GPUDepthStencilState,
} as GPURenderPipelineDescriptor);

//////////// RENDERPASS ////////////

const colorAttachment: GPURenderPassColorAttachment = {
    label: "color attachment",
    view: context!.getCurrentTexture().createView(),
    clearValue: [1, 1, 1, 1],
    loadOp: "clear",
    storeOp: "store",
} as GPURenderPassColorAttachment;
const depthTexture: GPUTexture = device.createTexture({
    label: "depth texture",
    size: [canvas.width, canvas.height],
    format: "depth32float",
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

const uniformArrayBuffer: ArrayBufferLike = uniformData.buffer;
const uniformBuffer: GPUBuffer = device.createBuffer({
    label: "uniforms uniform buffer",
    size: uniformArrayBuffer.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
} as GPUBufferDescriptor);
device!.queue.writeBuffer(uniformBuffer, 0, uniformArrayBuffer);

//////////// VERTICES ////////////

const data: OBJParseResult = await loadOBJ("./resources/bunny.obj");
// /*
const now: float = performance.now();
const {
    meshlets,
    clusters, // triangle indices per meshlet
    triangleAdjacency, // triangle adjacency
}: TriangleClusteringResult = await clusterizeTriangles({
    positions: data.vertices,
    indices: data.indices!,
});
log("clusterize", performance.now() - now, "ms");

const meshletToGroup: int[] = [];
const now2: float = performance.now();
const groups = await groupClusters(meshlets, clusters, triangleAdjacency);
log("group", performance.now() - now2, "ms");
const gStat: Uint32Array = new Uint32Array(100);
for (let g: int = 0; g < groups.length; g++) {
    gStat[groups[g].length]++;
    groups[g].forEach((meshletIndex) => (meshletToGroup[meshletIndex] = g));
}

const meshletsCount: int = meshlets.length;
const verticesCount: int = meshletsCount * maxTriangles * numVertices;
const verticesData: Float32Array = new Float32Array(
    verticesCount * vertexStride,
);
const stat: Uint32Array = new Uint32Array(1000);
for (let m: int = 0; m < meshletsCount; m++) {
    const group: int = meshletToGroup[m];
    const indices: Uint32Array = meshlets[m].indices;
    const vertices: Float32Array = meshlets[m].positions;
    const numTriangles: int = indices.length / 3;
    stat[numTriangles]++;
    for (let t: int = 0; t < numTriangles; t++) {
        const triangleOffset: int = m * maxTriangles + t;

        for (let v: int = 0; v < numVertices; v++) {
            const vertexOffset: int = triangleOffset * numVertices + v;
            const index: int = indices[t * numVertices + v];

            const srcOffset: int = index * vertexStride;
            const dstOffset: int = vertexOffset * vertexStride;
            verticesData[dstOffset + 0] = vertices[srcOffset + 0];
            verticesData[dstOffset + 1] = vertices[srcOffset + 1];
            verticesData[dstOffset + 2] = vertices[srcOffset + 2];
            verticesData[dstOffset + 3] = group;
        }
    }
}
log("cluster stats:");
for (let i: int = 0; i < stat.length; i++) {
    if (stat[i] !== 0) {
        log(i + ": " + stat[i]);
    }
}
log("group stats:");
for (let i: int = 0; i < gStat.length; i++) {
    if (gStat[i] !== 0) {
        log(i + ": " + gStat[i]);
    }
}
// */
/*
const meshlets = MeshoptClusterizer.buildMeshlets(
    data.indices!,
    data.vertices,
    vertexStride,
    maxVertices,
    maxTriangles,
);
const meshletsCount: int = meshlets.meshletCount;
const verticesCount: int = meshletsCount * maxTriangles * numVertices;
const verticesData: Float32Array = new Float32Array(
    verticesCount * vertexStride,
);
let stat: Uint32Array = new Uint32Array(maxTriangles + 1);
for (let i: int = 0; i < meshletsCount; i++) {
    const meshletsOffset: int = i * vertexStride;
    const vertexOffset: int = meshlets.meshlets[meshletsOffset + 0];
    const triangleOffset: int = meshlets.meshlets[meshletsOffset + 1];
    const triangleCount: int = meshlets.meshlets[meshletsOffset + 3];
    const targetOffset: int = i * maxTriangles * numVertices;
    stat[triangleCount]++;
    for (let j: int = 0; j < triangleCount * numVertices; j++) {
        const trianglesOffset: int = triangleOffset + j;
        const verticesIndex: int =
            vertexOffset + meshlets.triangles[trianglesOffset];
        const originalIndex: float =
            meshlets.vertices[verticesIndex] * vertexStride;
        const verticesOffset: int = (targetOffset + j) * vertexStride;
        verticesData[verticesOffset + 0] = data.vertices[originalIndex + 0];
        verticesData[verticesOffset + 1] = data.vertices[originalIndex + 1];
        verticesData[verticesOffset + 2] = data.vertices[originalIndex + 2];
    }
}
for (let i: int = 0; i < stat.length; i++) {
    if (stat[i] !== 0) {
        log(i + ": " + stat[i]);
    }
}
*/

const vertexArrayBuffer: ArrayBufferLike = verticesData.buffer;
const verticesBuffer: GPUBuffer = device.createBuffer({
    label: "vertex buffer",
    size: vertexArrayBuffer.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
} as GPUBufferDescriptor);
device.queue.writeBuffer(verticesBuffer, 0, vertexArrayBuffer);

log("meshlets", dotit(meshletsCount));
log("vertices", dotit(verticesCount));

//////////// INSTANCES ////////////

const instancesData: Float32Array = new Float32Array(
    instanceCount * instanceStride,
);
const uIntView: Uint32Array = new Uint32Array(instancesData.buffer);

for (let i: int = 0; i < instanceCount; i++) {
    new Vec3(
        Math.floor(i / (spawnSize * spawnSize)),
        Math.floor(i / spawnSize) % spawnSize,
        i % spawnSize,
    )
        .scale(3)
        .store(instancesData, i * instanceStride);
    uIntView[i * instanceStride + 3] = 0;
    uIntView[i * instanceStride + 4] = meshletsCount;
}

const instancesArrayBuffer: ArrayBufferLike = instancesData.buffer;
const instancesBuffer: GPUBuffer = device.createBuffer({
    label: "instances buffer",
    size: instancesArrayBuffer.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
} as GPUBufferDescriptor);
device.queue.writeBuffer(instancesBuffer, 0, instancesArrayBuffer);

log("instances", dotit(instanceCount));

//////////// TASKS ////////////

const tasksBuffer: GPUBuffer = device.createBuffer({
    label: "task buffer",
    size: maxTasks * taskStride * byteSize,
    usage: GPUBufferUsage.STORAGE,
} as GPUBufferDescriptor);

//////////// INDIRECT ////////////

const indirectData: Uint32Array = new Uint32Array([
    maxTriangles * numVertices,
    0,
    0,
    0,
]);
const indirectArrayBuffer: ArrayBufferLike = indirectData.buffer;
const indirectBuffer: GPUBuffer = device.createBuffer({
    label: "indirect buffer",
    size: 4 * byteSize,
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

const computeBindGroup: GPUBindGroup = device.createBindGroup({
    label: "compute bind group",
    layout: computePipeline.getBindGroupLayout(0),
    entries: [
        {
            binding: 0,
            resource: { buffer: instancesBuffer } as GPUBindingResource,
        } as GPUBindGroupEntry,
        {
            binding: 1,
            resource: { buffer: tasksBuffer } as GPUBindingResource,
        } as GPUBindGroupEntry,
        {
            binding: 2,
            resource: { buffer: indirectBuffer } as GPUBindingResource,
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
            resource: { buffer: instancesBuffer } as GPUBindingResource,
        } as GPUBindGroupEntry,
        {
            binding: 3,
            resource: { buffer: tasksBuffer } as GPUBindingResource,
        } as GPUBindGroupEntry,
    ],
} as GPUBindGroupDescriptor);

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
stats.set("gpu delta", 0);
stats.set("compute delta", 0);
stats.set("render delta", 0);
stats.set("meshlets", 0);
stats.show();
const frameDelta: RollingAverage = new RollingAverage(60);
const gpuDelta: RollingAverage = new RollingAverage(60);
const computeDelta: RollingAverage = new RollingAverage(60);
const renderDelta: RollingAverage = new RollingAverage(60);

//////////// RENDER BUNDLE ////////////

const renderBundleEncoder: GPURenderBundleEncoder =
    device.createRenderBundleEncoder({
        label: "render bundle",
        colorFormats: [presentationFormat],
        depthStencilFormat: "depth32float",
    } as GPURenderBundleEncoderDescriptor);
renderBundleEncoder.setPipeline(renderPipeline);
renderBundleEncoder.setBindGroup(0, renderBindGroup);
renderBundleEncoder.drawIndirect(indirectBuffer, 0);
const renderBundle: GPURenderBundle = renderBundleEncoder.finish();

//////////// RENDER FRAME ////////////

async function render(now: float): Promise<void> {
    //////////// UPDATE ////////////

    control.update();
    cameraView.view(cameraPos, cameraDir, up);
    viewProjection.multiply(cameraView, projection).store(uniformData, 0);
    device!.queue.writeBuffer(uniformBuffer, 0, uniformArrayBuffer);

    //////////// RENDER ////////////

    colorAttachment.view = context!.getCurrentTexture().createView();
    depthStencilAttachment.view = depthTexture.createView();
    device!.queue.writeBuffer(indirectBuffer, 0, indirectArrayBuffer);

    const renderEncoder: GPUCommandEncoder = device!.createCommandEncoder({
        label: "render command encoder",
    } as GPUObjectDescriptorBase);

    const computePass: GPUComputePassEncoder = renderEncoder.beginComputePass({
        label: "compute pass",
        timestampWrites: {
            querySet: querySet,
            beginningOfPassWriteIndex: 2,
            endOfPassWriteIndex: 3,
        } as GPURenderPassTimestampWrites,
    } as GPUComputePassDescriptor);
    computePass.setPipeline(computePipeline);
    computePass.setBindGroup(0, computeBindGroup);
    computePass.dispatchWorkgroups(instanceCount, 1, 1);
    computePass.end();

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
            stats.set("meshlets", result[1]);
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
                Number(
                    timingsNanoseconds[3] -
                        timingsNanoseconds[2] +
                        (timingsNanoseconds[1] - timingsNanoseconds[0]),
                ) / 1_000_000,
            );
            gpuDelta.sample(stats.get("gpu delta")!);
            stats.set(
                "compute delta",
                Number(timingsNanoseconds[3] - timingsNanoseconds[2]) /
                    1_000_000,
            );
            computeDelta.sample(stats.get("compute delta")!);
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
    // prettier-ignore
    stats.update(`
        <b>frame rate: ${(1_000 / frameDelta.get()).toFixed(0)} fps</b><br>
        frame delta: ${frameDelta.get().toFixed(2)} ms<br>
        <br>
        <b>gpu rate: ${(1_000 / gpuDelta.get()).toFixed(0)} fps</b><br>
        gpu delta: ${gpuDelta.get().toFixed(2)} ms<br>
        |- compute delta: ${computeDelta.get().toFixed(2)} ms<br>
        |- render delta: ${renderDelta.get().toFixed(2)} ms<br>
        <br>
        <b>meshlet count: ${stats.get("meshlets")}</b>
    `);
    stats.set("frame delta", now);

    requestAnimationFrame(render);
}
requestAnimationFrame(render);
