/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, October 2023
 */

import { int, float } from "../../types/utilities/utils.type.js";
import { Vec3 } from "../utilities/Vec3.js";

export type ChunkGeometryData = {
    positions: Float32Array;
    indicies: Uint32Array;
};

function generateGeometry(resolution: int): ChunkGeometryData {
    const positions: float[] = [];
    const indices: int[] = [];
    const side: int = resolution + 1;
    const scale: float = 1 / resolution;
    for (let y: int = 0; y < side; y++) {
        for (let x: int = 0; x < side; x++) {
            positions.push(x * scale - 0.5, 0, y * scale - 0.5, 0);
        }
    }
    for (let y: int = 0; y < resolution; y++) {
        for (let x: int = 0; x < resolution; x++) {
            const topLeft: int = y * side + x;
            const topRight: int = topLeft + 1;
            const bottomLeft: int = topLeft + side;
            const bottomRight: int = bottomLeft + 1;
            indices.push(topLeft, bottomLeft, topRight);
            indices.push(topRight, bottomLeft, bottomRight);
        }
    }
    return {
        positions: new Float32Array(positions),
        indicies: new Uint32Array(indices),
    };
}

export const ChunkResolution: int = 32;

export const ChunkGeometry: ChunkGeometryData =
    generateGeometry(ChunkResolution);

export const ChunkMinSize: float = 32;

export type ChunkData = {
    position: Vec3;
    size: float;
};

export const ChunkInstanceLength: int = 3 + 1;

export function storeChunk(
    instances: Float32Array,
    index: int,
    chunk: ChunkData,
): void {
    instances[index + 0] = chunk.position.x;
    instances[index + 1] = chunk.position.y;
    instances[index + 2] = chunk.position.z;
    instances[index + 3] = chunk.size;
}
