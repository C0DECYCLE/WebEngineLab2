/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, October 2023
 */

import {
    int,
    Nullable,
    Undefinable,
} from "../../types/utilities/utils.type.js";
import { assert } from "../utilities/utils.js";

export const byteSize: int = 4;

export function createCanvas(): HTMLCanvasElement {
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

export async function createGPU(): Promise<{
    canvas: HTMLCanvasElement;
    device: GPUDevice;
    context: GPUCanvasContext;
    presentationFormat: GPUTextureFormat;
}> {
    const canvas: HTMLCanvasElement = createCanvas();
    const adapter: Nullable<GPUAdapter> = await navigator.gpu?.requestAdapter();
    const device: Undefinable<GPUDevice> = await adapter?.requestDevice({
        requiredFeatures: ["timestamp-query"],
    } as GPUDeviceDescriptor);
    assert(device);
    const context: Nullable<GPUCanvasContext> = canvas.getContext("webgpu");
    assert(context);
    const presentationFormat: GPUTextureFormat =
        navigator.gpu.getPreferredCanvasFormat();
    context.configure({
        device: device,
        format: presentationFormat,
    } as GPUCanvasConfiguration);
    return { canvas, device, context, presentationFormat };
}

export async function loadShader(
    device: GPUDevice,
    url: string,
): Promise<GPUShaderModule> {
    return device.createShaderModule({
        label: url,
        code: await fetch(url).then(
            async (response: Response) => await response.text(),
        ),
    } as GPUShaderModuleDescriptor);
}
