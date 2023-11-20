/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, October 2023
 */

import { float, int } from "../utilities/utils.type.js";
import { byteSize } from "./helper.js";
import { log } from "../utilities/logger.js";
import { dotit } from "../utilities/utils.js";
import { Vec3 } from "../utilities/Vec3.js";
import { OBJParseResult, OBJParser } from "../OBJParser.js";
import { TreeSize } from "./tree.js";

export async function createCube(
    device: GPUDevice,
    presentationFormat: GPUTextureFormat,
    uniformBuffer: GPUBuffer,
): Promise<{ cubeBundle: GPURenderBundle }> {
    //////////// SETUP CUBE ////////////

    const raw: string = await fetch("./resources/cube.obj").then(
        async (response: Response) => await response.text(),
    );
    const parser: OBJParser = new OBJParser();
    const data: OBJParseResult = parser.parse(raw, true);

    //////////// SETUP VERTECIES ////////////

    const verteciesCount: int = data.positions.length / 4;
    const verteciesBuffer: GPUBuffer = device.createBuffer({
        label: "cube vertex buffer",
        size: data.positions.buffer.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    } as GPUBufferDescriptor);
    device.queue.writeBuffer(verteciesBuffer, 0, data.positions.buffer);

    log("cube vertecies", dotit(verteciesCount));

    //////////// SETUP INDICES ////////////

    const indicesCount: int = data.indicies!.length;
    const indicesBuffer: GPUBuffer = device.createBuffer({
        label: "cube index buffer",
        size: data.indicies!.buffer.byteLength,
        usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    } as GPUBufferDescriptor);
    device.queue.writeBuffer(indicesBuffer, 0, data.indicies!.buffer);

    log("cube indices", dotit(indicesCount));

    //////////// SETUP INSTANCES ////////////

    const every: int = 16;
    const n: int = TreeSize / every;
    const off: float = TreeSize * 0.5 - every * 0.5;
    const cubeInstanceFloats: int = 3 + 1;
    const cubeInstanceCount: int = n * n;
    const cubeInstancesData: Float32Array = new Float32Array(
        cubeInstanceCount * cubeInstanceFloats,
    );
    for (let y: int = 0; y < n; y++) {
        for (let x: int = 0; x < n; x++) {
            new Vec3(x * every, 0, y * every)
                .sub(off, 0, off)
                .store(cubeInstancesData, (y * n + x) * cubeInstanceFloats);
        }
    }
    const cubeInstancesBuffer: GPUBuffer = device.createBuffer({
        label: "terrain instances buffer",
        size: cubeInstancesData.buffer.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    } as GPUBufferDescriptor);
    device.queue.writeBuffer(cubeInstancesBuffer, 0, cubeInstancesData.buffer);

    //////////// SETUP INDIRECT ////////////

    const cubeIndirectData: Uint32Array = new Uint32Array([
        indicesCount,
        cubeInstanceCount,
        0,
        0,
        0,
    ]);
    const cubeIndirectBuffer: GPUBuffer = device.createBuffer({
        label: "cube indirect buffer",
        size: 5 * byteSize,
        usage:
            GPUBufferUsage.INDIRECT |
            GPUBufferUsage.STORAGE |
            GPUBufferUsage.COPY_SRC |
            GPUBufferUsage.COPY_DST,
    } as GPUBufferDescriptor);
    device.queue.writeBuffer(cubeIndirectBuffer, 0, cubeIndirectData.buffer);

    //////////// SETUP SHADER ////////////

    const cubeShader: GPUShaderModule = device.createShaderModule({
        label: "cube shader",
        code: await fetch("./shaders/terrain/cube.wgsl").then(
            async (response: Response) => await response.text(),
        ),
    } as GPUShaderModuleDescriptor);

    //////////// SETUP PIPELINE ////////////

    const cubePipeline: GPURenderPipeline =
        await device.createRenderPipelineAsync({
            label: "cube pipeline",
            layout: "auto",
            vertex: {
                module: cubeShader,
                entryPoint: "vs",
            } as GPUVertexState,
            fragment: {
                module: cubeShader,
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

    //////////// SETUP BINDGROUP ////////////

    const cubeBindGroup: GPUBindGroup = device.createBindGroup({
        label: "cube bind group",
        layout: cubePipeline.getBindGroupLayout(0),
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
                resource: {
                    buffer: cubeInstancesBuffer,
                } as GPUBindingResource,
            } as GPUBindGroupEntry,
        ],
    } as GPUBindGroupDescriptor);

    //////////// SETUP RENDER BUNDLE ////////////

    const cubeBundleEncoder: GPURenderBundleEncoder =
        device.createRenderBundleEncoder({
            label: "cube bundle",
            colorFormats: [presentationFormat],
            depthStencilFormat: "depth24plus",
        } as GPURenderBundleEncoderDescriptor);
    cubeBundleEncoder.setPipeline(cubePipeline);
    cubeBundleEncoder.setBindGroup(0, cubeBindGroup);
    cubeBundleEncoder.setIndexBuffer(indicesBuffer, "uint32");
    cubeBundleEncoder.drawIndexedIndirect(cubeIndirectBuffer, 0);
    const cubeBundle: GPURenderBundle = cubeBundleEncoder.finish();

    return { cubeBundle };
}
