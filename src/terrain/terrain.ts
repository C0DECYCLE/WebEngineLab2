/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, October 2023
 */

import { int } from "../../types/utilities/utils.type.js";
import {
    ChunkGeometry,
    ChunkGeometryData,
    ChunkInstanceLength,
} from "./chunk.js";
import { byteSize } from "./helper.js";
import { log } from "../utilities/logger.js";
import { dotit } from "../utilities/utils.js";
import {
    MaxTreeLength,
    TreeData,
    TreeSize,
    generateTree,
    storeTree,
} from "./tree.js";
import { Vec3 } from "../utilities/Vec3.js";

export async function createTerrain(
    device: GPUDevice,
    presentationFormat: GPUTextureFormat,
    uniformBuffer: GPUBuffer,
): Promise<{
    terrainBundle: GPURenderBundle;
    terrainInstancesData: Float32Array;
    terrainInstancesBuffer: GPUBuffer;
    terrainIndirectData: Uint32Array;
    terrainIndirectBuffer: GPUBuffer;
}> {
    const data: ChunkGeometryData = ChunkGeometry;

    //////////// SETUP VERTICES ////////////

    const verticesCount: int = data.positions.length / 4;
    const verticesBuffer: GPUBuffer = device.createBuffer({
        label: "terrain vertex buffer",
        size: data.positions.buffer.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    } as GPUBufferDescriptor);
    device.queue.writeBuffer(verticesBuffer, 0, data.positions.buffer);

    log("terrain vertices", dotit(verticesCount));

    //////////// SETUP INDICES ////////////

    const indicesCount: int = data.indicies.length;
    const indicesBuffer: GPUBuffer = device.createBuffer({
        label: "terrain index buffer",
        size: data.indicies.buffer.byteLength,
        usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    } as GPUBufferDescriptor);
    device.queue.writeBuffer(indicesBuffer, 0, data.indicies.buffer);

    log("terrain indices", dotit(indicesCount));

    //////////// SETUP INSTANCES ////////////

    const terrainInstancesData: Float32Array = new Float32Array(
        MaxTreeLength * ChunkInstanceLength,
    );
    const terrainInstancesBuffer: GPUBuffer = device.createBuffer({
        label: "terrain instances buffer",
        size: terrainInstancesData.buffer.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    } as GPUBufferDescriptor);

    //////////// SETUP INDIRECT ////////////

    const terrainIndirectData: Uint32Array = new Uint32Array([
        indicesCount,
        0, //instanceCount,
        0,
        0,
        0,
    ]);
    const terrainIndirectBuffer: GPUBuffer = device.createBuffer({
        label: "terrain indirect buffer",
        size: 5 * byteSize,
        usage:
            GPUBufferUsage.INDIRECT |
            GPUBufferUsage.STORAGE |
            GPUBufferUsage.COPY_SRC |
            GPUBufferUsage.COPY_DST,
    } as GPUBufferDescriptor);

    //////////// SETUP SHADER ////////////

    const terrainShader: GPUShaderModule = device.createShaderModule({
        label: "terrain shader",
        code: await fetch("./shaders/terrain/terrain.wgsl").then(
            async (response: Response) => await response.text(),
        ),
    } as GPUShaderModuleDescriptor);

    //////////// SETUP PIPELINE ////////////

    const terrainPipeline: GPURenderPipeline =
        await device.createRenderPipelineAsync({
            label: "terrain pipeline",
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

    //////////// SETUP BINDGROUP ////////////

    const terrainBindGroup: GPUBindGroup = device.createBindGroup({
        label: "terrain bind group",
        layout: terrainPipeline.getBindGroupLayout(0),
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
                resource: {
                    buffer: terrainInstancesBuffer,
                } as GPUBindingResource,
            } as GPUBindGroupEntry,
        ],
    } as GPUBindGroupDescriptor);

    //////////// SETUP RENDER BUNDLE ////////////

    const terrainBundleEncoder: GPURenderBundleEncoder =
        device.createRenderBundleEncoder({
            label: "terrain bundle",
            colorFormats: [presentationFormat],
            depthStencilFormat: "depth24plus",
        } as GPURenderBundleEncoderDescriptor);
    terrainBundleEncoder.setPipeline(terrainPipeline);
    terrainBundleEncoder.setBindGroup(0, terrainBindGroup);
    terrainBundleEncoder.setIndexBuffer(indicesBuffer, "uint32");
    terrainBundleEncoder.drawIndexedIndirect(terrainIndirectBuffer, 0);
    const terrainBundle: GPURenderBundle = terrainBundleEncoder.finish();

    return {
        terrainBundle,
        terrainInstancesData,
        terrainInstancesBuffer,
        terrainIndirectData,
        terrainIndirectBuffer,
    };
}

export function updateTerrain(
    device: GPUDevice,
    cameraPos: Vec3,
    terrainInstancesData: Float32Array,
    terrainInstancesBuffer: GPUBuffer,
    terrainIndirectData: Uint32Array,
    terrainIndirectBuffer: GPUBuffer,
): void {
    //////////// UPDATE TREE ////////////

    const tree: TreeData = generateTree(new Vec3(), TreeSize, cameraPos);

    //////////// UPDATE INSTANCES ////////////

    const instancesCount: int = storeTree(terrainInstancesData, tree);
    device.queue.writeBuffer(
        terrainInstancesBuffer,
        0,
        terrainInstancesData.buffer,
        0,
        instancesCount * ChunkInstanceLength * byteSize,
    );

    //////////// UPDATE INDIRECT ////////////

    terrainIndirectData[1] = instancesCount;
    device!.queue.writeBuffer(
        terrainIndirectBuffer,
        0,
        terrainIndirectData.buffer,
    );
}
