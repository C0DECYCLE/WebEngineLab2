/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, October 2023
 */

import { OBJParseResult, OBJParser } from "../OBJParser.js";
import { Vec3 } from "../utilities/Vec3.js";
import { log } from "../utilities/logger.js";
import { dotit } from "../utilities/utils.js";
import { int } from "../utilities/utils.type.js";

export class Geometry {
    public constructor(filePath: string, instances: int) {
        //
    }

    public async construct(): Promise<void> {
        //
    }

    /*
    private readonly indirectData: Uint32Array;
    private readonly instanceData: Float32Array;
    private vertexBuffer: GPUBuffer;
    private indexBuffer: GPUBuffer;
    private instanceBuffer: GPUBuffer;
    private indirectBuffer: GPUBuffer;
    private bindGroup: GPUBindGroup;
    public bundle: GPURenderBundle;

    public constructor(
        device: GPUDevice,
        filePath: string,
        instances: int,
        uniformBuffer: GPUBuffer,
        pipeline: GPURenderPipeline,
    ) {
        this.indirectData = new Uint32Array([0, 0, 0, 0, 0]);
        this.instanceData = this.createInstanceData(filePath, instances);
        this.load(filePath).then(async (data: OBJParseResult) => {
            this.vertexBuffer = this.createVertecies(
                device,
                filePath,
                data.positions,
            );
            this.indexBuffer = this.createIndices(
                device,
                filePath,
                data.indices!,
            );
            this.instanceBuffer = this.createInstances(device, filePath);
            this.indirectBuffer = this.createIndirect(device, filePath);
            this.bindGroup = this.createBindGroup(
                device,
                filePath,
                uniformBuffer,
                pipeline,
            );
            this.bundle = this.createBundle(device, filePath, pipeline);
        });
    }

    private createInstanceData(filePath: string, count: int): Float32Array {
        const attr: int = 3 + 1;
        const floats: int = attr + attr;
        const data: Float32Array = new Float32Array(count * floats);
        for (let i: int = 0; i < count; i++) {
            new Vec3(Math.random(), Math.random(), Math.random())
                .sub(0.5)
                .scale(Math.cbrt(count) * 10)
                .store(data, i * floats);
            new Vec3(
                (filePath.charCodeAt(0) * 0.323) % 1,
                (filePath.charCodeAt(1) * 0.116) % 1,
                (filePath.charCodeAt(2) * 0.735) % 1,
            ).store(data, i * floats + attr);
        }
        this.indirectData[1] = count;
        log(`${filePath} instances`, dotit(count));
        return data;
    }

    private async load(filePath: string): Promise<OBJParseResult> {
        const raw: string = await fetch(`./resources/${filePath}`).then(
            async (response: Response) => await response.text(),
        );
        return OBJParser.Standard.parse(raw, true);
    }

    private createVertecies(
        device: GPUDevice,
        filePath: string,
        positions: Float32Array,
    ): GPUBuffer {
        const buffer: GPUBuffer = device.createBuffer({
            label: `${filePath} vertex buffer`,
            size: positions.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        } as GPUBufferDescriptor);
        device.queue.writeBuffer(buffer, 0, positions.buffer);
        const count: int = positions.length / 4;
        log(`${filePath} vertecies`, dotit(count));
        return buffer;
    }

    private createIndices(
        device: GPUDevice,
        filePath: string,
        indices: Uint32Array,
    ): GPUBuffer {
        const buffer: GPUBuffer = device.createBuffer({
            label: `${filePath} index buffer`,
            size: indices.byteLength,
            usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
        } as GPUBufferDescriptor);
        device.queue.writeBuffer(buffer, 0, indices.buffer);
        const count: int = indices.length;
        this.indirectData[0] = count;
        log(`${filePath} indices`, dotit(count));
        return buffer;
    }

    private createInstances(device: GPUDevice, filePath: string): GPUBuffer {
        const buffer: GPUBuffer = device.createBuffer({
            label: `${filePath} instance buffer`,
            size: this.instanceData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        } as GPUBufferDescriptor);
        device.queue.writeBuffer(buffer, 0, this.instanceData.buffer);
        return buffer;
    }

    private createIndirect(device: GPUDevice, filePath: string): GPUBuffer {
        const buffer: GPUBuffer = device.createBuffer({
            label: `${filePath} indirect buffer`,
            size: this.indirectData.byteLength,
            usage:
                GPUBufferUsage.INDIRECT |
                GPUBufferUsage.STORAGE |
                GPUBufferUsage.COPY_SRC |
                GPUBufferUsage.COPY_DST,
        } as GPUBufferDescriptor);
        device.queue.writeBuffer(buffer, 0, this.indirectData.buffer);
        return buffer;
    }

    private createBindGroup(
        device: GPUDevice,
        filePath: string,
        uniformBuffer: GPUBuffer,
        pipeline: GPURenderPipeline,
    ): GPUBindGroup {
        return device.createBindGroup({
            label: `${filePath} bindgroup`,
            layout: pipeline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: { buffer: uniformBuffer } as GPUBindingResource,
                } as GPUBindGroupEntry,
                {
                    binding: 1,
                    resource: {
                        buffer: this.vertexBuffer,
                    } as GPUBindingResource,
                } as GPUBindGroupEntry,
                {
                    binding: 2,
                    resource: {
                        buffer: this.instanceBuffer,
                    } as GPUBindingResource,
                } as GPUBindGroupEntry,
            ],
        } as GPUBindGroupDescriptor);
    }

    private createBundle(
        device: GPUDevice,
        filePath: string,
        pipeline: GPURenderPipeline,
    ): GPURenderBundle {
        const presentationFormat: GPUTextureFormat =
            navigator.gpu.getPreferredCanvasFormat();
        const encoder: GPURenderBundleEncoder =
            device.createRenderBundleEncoder({
                label: `${filePath} bundle`,
                colorFormats: [presentationFormat],
                depthStencilFormat: "depth24plus",
            } as GPURenderBundleEncoderDescriptor);
        encoder.setPipeline(pipeline);
        encoder.setBindGroup(0, this.bindGroup);
        encoder.setIndexBuffer(this.indexBuffer, "uint32");
        encoder.drawIndexedIndirect(this.indirectBuffer, 0);
        return encoder.finish();
    }
    */
}
