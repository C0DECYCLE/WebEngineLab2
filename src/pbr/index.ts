/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger
 */

import { WebGPUSinglePassDownsampler } from "../../node_modules/webgpu-spd/dist/index.js";
import { Controller } from "../Controller.js";
import { RollingAverage } from "../RollingAverage.js";
import { Stats } from "../Stats.js";
import { Mat4 } from "../utilities/Mat4.js";
import { assert, toRadian } from "../utilities/utils.js";
import { float, int, Nullable, Undefinable } from "../utilities/utils.type.js";
import { Vec3 } from "../utilities/Vec3.js";
import { includeExternal, loadOBJ, loadTexture, OBJ } from "./helper.js";

//////////// CONSTS ////////////

const byteSize: int = 4;
const directory: string = "./resources/lantern/";
export const imageFormat: GPUTextureFormat = "rgba8unorm";
const depthFormat: GPUTextureFormat = "depth32float";

//////////// SETUP ////////////

const presentationFormat: GPUTextureFormat =
    navigator.gpu.getPreferredCanvasFormat();
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
const cameraData: Float32Array = new Float32Array(4 + 4 * 4);
const cameraBuffer: GPUBuffer = device.createBuffer({
    size: cameraData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(cameraBuffer, 0, cameraData.buffer);
const control: Controller = new Controller(canvas, {
    position: cameraPos,
    direction: cameraDir,
});

//////////// STATS ////////////

const deltaNames: string[] = ["render"];
const deltas: Map<string, RollingAverage> = new Map<string, RollingAverage>();
const stats: Stats = new Stats();
deltas.set("frame", new RollingAverage(60));
stats.set("frame" + " delta", 0);
deltas.set("gpu", new RollingAverage(60));
stats.set("gpu" + " delta", 0);
for (const name of deltaNames) {
    deltas.set(name, new RollingAverage(60));
    stats.set(name + " delta", 0);
}
stats.show();

//////////// GEOMETRY ////////////

const geometry: OBJ = await loadOBJ(directory + "lantern_full.obj");
const vertexBuffer: GPUBuffer = device.createBuffer({
    size: geometry.vertices.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(vertexBuffer, 0, geometry.vertices.buffer);
const indexBuffer: GPUBuffer = device.createBuffer({
    size: geometry.indices.byteLength,
    usage:
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.INDEX,
});
device.queue.writeBuffer(indexBuffer, 0, geometry.indices.buffer);

//////////// TEXTURE ////////////

const downsampler: WebGPUSinglePassDownsampler =
    new WebGPUSinglePassDownsampler({
        device: device,
        formats: [{ format: imageFormat }],
    });
const textureSampler: GPUSampler = device.createSampler({
    minFilter: "linear",
    magFilter: "linear",
    mipmapFilter: "nearest",
});
const baseColorTexture: GPUTexture = await loadTexture(
    device,
    downsampler,
    directory + "lantern_baseColor.jpg",
);
const normalTexture: GPUTexture = await loadTexture(
    device,
    downsampler,
    directory + "lantern_normal.jpg",
);
const specularTexture: GPUTexture = await loadTexture(
    device,
    downsampler,
    directory + "lantern_specular.jpg",
);
const glossTexture: GPUTexture = await loadTexture(
    device,
    downsampler,
    directory + "lantern_gloss.jpg",
);
const ambientOcclusionTexture: GPUTexture = await loadTexture(
    device,
    downsampler,
    directory + "lantern_ambientOcclusion.jpg",
);
const cavityTexture: GPUTexture = await loadTexture(
    device,
    downsampler,
    directory + "lantern_cavity.jpg",
);

//////////// GPU TIMING ////////////

const capacity: int = deltaNames.length * 2;
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

const renderShader: GPUShaderModule = device.createShaderModule({
    code: await includeExternal("render.wgsl"),
});

//////////// PIPELINE ////////////

const renderPipeline: GPURenderPipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: {
        module: renderShader,
        entryPoint: "vs",
    },
    fragment: {
        module: renderShader,
        entryPoint: "fs",
        targets: [{ format: presentationFormat }],
    },
    primitive: {
        cullMode: "back",
    },
    depthStencil: {
        depthWriteEnabled: true,
        depthCompare: "less",
        format: depthFormat,
    } as GPUDepthStencilState,
});

//////////// DEPTH ////////////

const depthTexture: GPUTexture = device.createTexture({
    size: [canvas.width, canvas.height],
    format: depthFormat,
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
});
const depthTextureView: GPUTextureView = depthTexture.createView();

//////////// BINDGROUP ////////////

const renderBindGroup: GPUBindGroup = device.createBindGroup({
    layout: renderPipeline.getBindGroupLayout(0),
    entries: [
        { binding: 0, resource: cameraBuffer },
        { binding: 1, resource: vertexBuffer },
        { binding: 2, resource: textureSampler },
        { binding: 3, resource: baseColorTexture },
        { binding: 4, resource: normalTexture },
        { binding: 5, resource: specularTexture },
        { binding: 6, resource: glossTexture },
        { binding: 7, resource: ambientOcclusionTexture },
        { binding: 8, resource: cavityTexture },
    ],
});

//////////// RENDER FRAME ////////////

function frame(now: float): void {
    assert(context && device);

    //////////// UPDATE ////////////

    control.update();
    cameraPos.store(cameraData, 0);
    cameraView.view(cameraPos, cameraDir, up);
    viewProjection.multiply(cameraView, projection).store(cameraData, 4);
    device.queue.writeBuffer(cameraBuffer, 0, cameraData.buffer);

    //////////// ENCODE ////////////

    const target: GPUTextureView = context.getCurrentTexture().createView();

    const encoder: GPUCommandEncoder = device.createCommandEncoder();

    const renderPass: GPURenderPassEncoder = encoder.beginRenderPass({
        colorAttachments: [
            {
                view: target,
                clearValue: [1, 1, 1, 1],
                loadOp: "clear",
                storeOp: "store",
            },
        ],
        depthStencilAttachment: {
            view: depthTextureView,
            depthClearValue: 1,
            depthLoadOp: "clear",
            depthStoreOp: "store",
        },
        timestampWrites: {
            querySet: querySet,
            beginningOfPassWriteIndex: 0,
            endOfPassWriteIndex: 1,
        },
    });
    renderPass.setPipeline(renderPipeline);
    renderPass.setBindGroup(0, renderBindGroup);
    renderPass.setIndexBuffer(indexBuffer, "uint32");
    renderPass.drawIndexed(geometry.indices.length);
    renderPass.end();

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
            const arrayBuffer: ArrayBuffer = queryReadbackBuffer
                .getMappedRange()
                .slice(0);
            queryReadbackBuffer.unmap();
            const timingsNanoseconds: BigInt64Array = new BigInt64Array(
                arrayBuffer,
            );
            const timings: float[] = Array.from(timingsNanoseconds).map(
                (value: bigint) => Number(value) / 1_000_000,
            );
            let min: float = Math.min(...timings);
            let max: float = Math.max(...timings);
            stats.set("gpu" + " delta", max - min);
            deltas.get("gpu")!.sample(stats.get("gpu" + " delta")!);
            for (let i: int = 0; i < deltaNames.length; i++) {
                const name: string = deltaNames[i];
                const a: float = timings[i * 2 + 0];
                const b: float = timings[i * 2 + 1];
                stats.set(name + " delta", b - a);
                deltas.get(name)!.sample(stats.get(name + " delta")!);
            }
        });
    }

    //////////// STATS ////////////

    stats.set("frame delta", now - stats.get("frame delta")!);
    deltas.get("frame")!.sample(stats.get("frame delta")!);
    // prettier-ignore
    stats.update(`
        <b>frame rate: ${(1_000 / deltas.get("frame")!.get()).toFixed(0)} fps</b><br>
        frame delta: ${deltas.get("frame")!.get().toFixed(2)} ms<br>
        gpu delta: ${deltas.get("gpu")!.get().toFixed(2)} ms<br>
        `);
    stats.set("frame delta", now);

    requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
