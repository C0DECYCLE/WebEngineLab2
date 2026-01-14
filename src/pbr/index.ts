/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger
 */

import { WebGPUSinglePassDownsampler } from "../../node_modules/webgpu-spd/dist/index.js";
import { Controller } from "../Controller.js";
import { RollingAverage } from "../RollingAverage.js";
import { Stats } from "../Stats.js";
import { log } from "../utilities/logger.js";
import { Mat4 } from "../utilities/Mat4.js";
import { assert, dotit, toRadian } from "../utilities/utils.js";
import { float, int, Nullable, Undefinable } from "../utilities/utils.type.js";
import { Vec3 } from "../utilities/Vec3.js";
import { includeExternal, loadOBJ, createTexture, OBJ } from "./helper.js";

//////////// CONSTS ////////////

export const byteSize: int = 4;
export const imageSize: int = 4096;
/*
- lantern
- suitcase
- desertcliff
- nordicrock
- statue
- bust
- snow
*/
const mesh: string = "snow";
const directory: string = "./resources/" + mesh + "/";
const depthFormat: GPUTextureFormat = "depth32float";
const linearFormat: GPUTextureFormat = "rgba8unorm";
const srgbFormat: GPUTextureFormat = "rgba8unorm-srgb";
(window as any).PBR = true;

//////////// SETUP ////////////

const swapchainFormat: GPUTextureFormat =
    navigator.gpu.getPreferredCanvasFormat();
const swapchainViewFormat: GPUTextureFormat = (swapchainFormat +
    "-srgb") as GPUTextureFormat;
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
    requiredFeatures: ["subgroups", "timestamp-query"],
});
const context: Nullable<GPUCanvasContext> = canvas.getContext("webgpu");
if (!device || !context) {
    throw new Error("Browser doesn't support WebGPU.");
}
context.configure({
    device: device,
    format: swapchainFormat,
    viewFormats: [swapchainViewFormat],
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

const gPre: float = performance.now();
const geometry: OBJ = await loadOBJ(directory + mesh + ".obj");
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
log("geometry", dotit(performance.now() - gPre), "ms");
log("vertices", dotit(geometry.vertices.length / 8));
log("triangles", dotit(geometry.indices.length / 3));

//////////// TEXTURE ////////////

/*
- baseColor / albedo 
- normal 
- specular 
- roughness 
- metalness
- Ambient occlusion 
- cavity 
- fuzz 
*/

const tPre: float = performance.now();
const downsampler: WebGPUSinglePassDownsampler =
    new WebGPUSinglePassDownsampler({
        device: device,
        formats: [{ format: linearFormat }],
    });
const textureSampler: GPUSampler = device.createSampler({
    minFilter: "linear",
    magFilter: "linear",
    mipmapFilter: "linear", // "nearest" for performance
});
const baseColorTexture: GPUTextureView = await createTexture(
    device,
    downsampler,
    directory + mesh + "_baseColor.jpg",
    linearFormat,
    srgbFormat,
);
const normalTexture: GPUTextureView = await createTexture(
    device,
    downsampler,
    directory + mesh + "_normal.jpg",
    linearFormat,
    linearFormat,
);
const specularTexture: GPUTextureView = await createTexture(
    device,
    downsampler,
    directory + mesh + "_specular.jpg",
    linearFormat,
    srgbFormat, //srgbFormat,
);
const roughnessTexture: GPUTextureView = await createTexture(
    device,
    downsampler,
    directory + mesh + "_roughness.jpg",
    linearFormat,
    linearFormat,
);
const metalnessTexture: GPUTextureView = await createTexture(
    device,
    downsampler,
    directory + mesh + "_metalness.jpg",
    linearFormat,
    linearFormat,
);
const ambientOcclusionTexture: GPUTextureView = await createTexture(
    device,
    downsampler,
    directory + mesh + "_ambientOcclusion.jpg",
    linearFormat,
    linearFormat,
);
const cavityTexture: GPUTextureView = await createTexture(
    device,
    downsampler,
    directory + mesh + "_cavity.jpg",
    linearFormat,
    srgbFormat, //srgbFormat,
);
const fuzzTexture: GPUTextureView = await createTexture(
    device,
    downsampler,
    directory + mesh + "_fuzz.jpg",
    linearFormat,
    linearFormat, //srgbFormat,
);
log("textures", dotit(performance.now() - tPre), "ms");

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

const nonPBRShader: GPUShaderModule = device.createShaderModule({
    code: await includeExternal("nonPBR.wgsl"),
});
const PBRShader: GPUShaderModule = device.createShaderModule({
    code: await includeExternal("PBR.wgsl"),
});

//////////// PIPELINE ////////////

const nonPBRPipeline: GPURenderPipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: {
        module: nonPBRShader,
        entryPoint: "vs",
    },
    fragment: {
        module: nonPBRShader,
        entryPoint: "fs",
        targets: [{ format: swapchainViewFormat }],
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
const PBRPipeline: GPURenderPipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: {
        module: PBRShader,
        entryPoint: "vs",
    },
    fragment: {
        module: PBRShader,
        entryPoint: "fs",
        targets: [{ format: swapchainViewFormat }],
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

const nonPBRBindGroup: GPUBindGroup = device.createBindGroup({
    layout: nonPBRPipeline.getBindGroupLayout(0),
    entries: [
        { binding: 0, resource: cameraBuffer },
        { binding: 1, resource: vertexBuffer },
        { binding: 2, resource: textureSampler },

        { binding: 3, resource: baseColorTexture },
        { binding: 4, resource: normalTexture },
        { binding: 5, resource: specularTexture },
        { binding: 6, resource: roughnessTexture },
        { binding: 7, resource: ambientOcclusionTexture },
        { binding: 8, resource: cavityTexture },
    ],
});
const PBRBindGroup: GPUBindGroup = device.createBindGroup({
    layout: PBRPipeline.getBindGroupLayout(0),
    entries: [
        { binding: 0, resource: cameraBuffer },
        { binding: 1, resource: vertexBuffer },
        { binding: 2, resource: textureSampler },

        { binding: 3, resource: baseColorTexture },
        { binding: 4, resource: normalTexture },
        { binding: 5, resource: roughnessTexture },
        { binding: 6, resource: metalnessTexture },
    ],
});

//////////// RENDER FRAME ////////////

function frame(now: float): void {
    assert(context && device);

    //////////// UPDATE ////////////

    control.update();
    cameraPos.store(cameraData, 0);
    cameraData[3] = now;
    cameraView.view(cameraPos, cameraDir, up);
    viewProjection.multiply(cameraView, projection).store(cameraData, 4);
    device.queue.writeBuffer(cameraBuffer, 0, cameraData.buffer);

    //////////// ENCODE ////////////

    const target: GPUTextureView = context.getCurrentTexture().createView({
        format: swapchainViewFormat,
    });

    const encoder: GPUCommandEncoder = device.createCommandEncoder();

    const renderPipeline: GPURenderPipeline = (window as any).PBR
        ? PBRPipeline
        : nonPBRPipeline;
    const renderBindGroup: GPUBindGroup = (window as any).PBR
        ? PBRBindGroup
        : nonPBRBindGroup;
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
