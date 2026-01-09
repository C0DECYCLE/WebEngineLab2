/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger
 */

import {
    maxMipLevelCount,
    SPDFilters,
    WebGPUSinglePassDownsampler,
} from "../../node_modules/webgpu-spd/dist/index.js";
import { assert } from "../utilities/utils.js";
import { float, int, Nullable } from "../utilities/utils.type.js";
import { Vec2 } from "../utilities/Vec2.js";
import { Vec3 } from "../utilities/Vec3.js";
import { imageFormat } from "./index.js";

//////////// SHADER ////////////

const directory: string = "./shaders/pbr/";
const includesDirectory: string = "./shaders/pbr/includes/";
const key: string = "#include";

export async function includeExternal(file: string): Promise<string> {
    const source: string = await (await fetch(directory + file)).text();
    const lines: string[] = source.split("\n");
    for (let i: int = 0; i < lines.length; i++) {
        const line: string = lines[i];
        if (!line.startsWith(key)) {
            continue;
        }
        const subfile: string = line.split(key)[1].trim().split(";")[0];
        lines[i] = await (await fetch(includesDirectory + subfile)).text();
    }
    return lines.join("\n");
}

//////////// GEOMETRY ////////////

export type OBJ = {
    vertices: Float32Array;
    indices: Uint32Array;
};

export async function loadOBJ(file: string): Promise<OBJ> {
    const lines: string[] = (await (await fetch(file)).text()).split("\n");
    const positions: Vec3[] = [];
    const normals: Vec3[] = [];
    const uvs: Vec2[] = [];
    const vertices: float[] = [];
    const indices: int[] = [];
    const cache: Map<string, int> = new Map<string, int>();
    let next: int = 0;
    for (const line of lines) {
        const parts: string[] = line.trim().split(" ");
        if (parts[0] === "v") {
            const x: float = parseFloat(parts[1]);
            const y: float = parseFloat(parts[2]);
            const z: float = parseFloat(parts[3]);
            positions.push(new Vec3(x, y, z));
        }
        if (parts[0] === "vn") {
            const x: float = parseFloat(parts[1]);
            const y: float = parseFloat(parts[2]);
            const z: float = parseFloat(parts[3]);
            normals.push(new Vec3(x, y, z).normalize());
        }
        if (parts[0] === "vt") {
            const u: float = parseFloat(parts[1]);
            const v: float = parseFloat(parts[2]);
            uvs.push(new Vec2(u, v));
        }
        if (parts[0] === "f") {
            const face: string[] = [parts[1], parts[2], parts[3]];
            for (const vertex of face) {
                if (!cache.get(vertex)) {
                    const subparts: string[] = vertex.trim().split("/");
                    const position: int = parseInt(subparts[0]) - 1;
                    const normal: int = parseInt(subparts[2]) - 1;
                    const uv: int = parseInt(subparts[1]) - 1;
                    vertices.push(...positions[position].toArray());
                    vertices.push(...normals[normal].toArray());
                    vertices.push(...uvs[uv].toArray());
                    cache.set(vertex, next++);
                }
                const index: Nullable<int> = cache.get(vertex) ?? null;
                assert(index !== null);
                indices.push(index);
            }
        }
    }
    const obj: OBJ = {
        vertices: new Float32Array(vertices),
        indices: new Uint32Array(indices),
    };
    return obj;
}

//////////// TEXTURE ////////////

export async function loadTexture(
    device: GPUDevice,
    downsampler: WebGPUSinglePassDownsampler,
    file: string,
): Promise<GPUTexture> {
    const blob: Blob = await (await fetch(file)).blob();
    const imageBitmap: ImageBitmap = await createImageBitmap(blob);
    const texture: GPUTexture = device.createTexture({
        format: imageFormat,
        size: [imageBitmap.width, imageBitmap.height],
        mipLevelCount: maxMipLevelCount(imageBitmap.width, imageBitmap.height),
        usage:
            GPUTextureUsage.TEXTURE_BINDING |
            GPUTextureUsage.STORAGE_BINDING |
            GPUTextureUsage.COPY_DST |
            GPUTextureUsage.RENDER_ATTACHMENT,
    });
    device.queue.copyExternalImageToTexture(
        { source: imageBitmap, flipY: true },
        { texture: texture, mipLevel: 0 },
        [imageBitmap.width, imageBitmap.height],
    );
    downsampler.generateMipmaps(device, texture, {
        filter: SPDFilters.Average,
    });
    return texture;
}
