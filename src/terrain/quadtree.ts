/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, October 2023
 */

import { int, float } from "../../types/utilities/utils.type.js";
import { Vec3 } from "../utilities/Vec3.js";
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

export function treeToInstances(tree: TreeData): Float32Array {
    const instances: Float32Array = new Float32Array(
        tree.length * ChunkInstanceLength,
    );
    for (let i: int = 0; i < tree.length; i++) {
        storeChunk(instances, i * ChunkInstanceLength, tree[i]);
    }
    return instances;
}
