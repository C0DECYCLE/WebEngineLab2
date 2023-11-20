/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, October 2023
 */

import { int, float } from "../utilities/utils.type.js";
import { Vec3 } from "../utilities/Vec3.js";
import { warn } from "../utilities/logger.js";
import {
    ChunkData,
    ChunkInstanceLength,
    ChunkMinSize,
    storeChunk,
} from "./chunk.js";

export type TreeData = ChunkData[];

export function generateTree(origin: Vec3, size: float, point: Vec3): TreeData {
    if (
        size > ChunkMinSize &&
        point.clone().sub(origin).lengthQuadratic() < size * size
    ) {
        const half: float = size / 2;
        const quad: float = half / 2;
        return [
            ...generateTree(origin.clone().add(-quad, 0, -quad), half, point),
            ...generateTree(origin.clone().add(quad, 0, -quad), half, point),
            ...generateTree(origin.clone().add(-quad, 0, quad), half, point),
            ...generateTree(origin.clone().add(quad, 0, quad), half, point),
        ];
    }
    return [{ position: origin, size: size } as ChunkData];
}

export const MaxTreeLength: int = 40;

export function storeTree(instances: Float32Array, tree: TreeData): int {
    if (tree.length > MaxTreeLength) {
        warn(`Ran out of tree length. (${tree.length} -> ${MaxTreeLength})`);
    }
    const length: int = Math.min(tree.length, MaxTreeLength);
    for (let i: int = 0; i < length; i++) {
        storeChunk(instances, i * ChunkInstanceLength, tree[i]);
    }
    return length;
}
