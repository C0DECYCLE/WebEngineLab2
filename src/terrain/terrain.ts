/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, October 2023
 */

import { byteSize, loadShader } from "./helper.js";

//////////// SHADER ////////////

export async function createTerrainShader(
    device: GPUDevice,
): Promise<GPUShaderModule> {
    return await loadShader(device, "./shaders/terrain/terrain.wgsl");
}
