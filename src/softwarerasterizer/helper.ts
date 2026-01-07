/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger
 */

import { OBJParseResult } from "../OBJParser.js";
import { assert } from "../utilities/utils.js";
import { float, int } from "../utilities/utils.type.js";

//////////// VOXEL ////////////

type Vec3 = [number, number, number];

export function voxelizeOBJ(
    obj: OBJParseResult,
    voxelsPerMeter: number,
): Float32Array {
    assert(obj.indices && obj.indicesCount);
    const stride = 8;
    const vStride = 8; // input vertex stride
    const voxelSize = voxelsPerMeter;

    const vertices = obj.vertices;
    const indices = obj.indices;

    // --- compute mesh AABB ---
    let min: Vec3 = [Infinity, Infinity, Infinity];
    let max: Vec3 = [-Infinity, -Infinity, -Infinity];

    for (let i = 0; i < obj.verticesCount; i++) {
        const o = i * vStride;
        const x = vertices[o];
        const y = vertices[o + 1];
        const z = vertices[o + 2];

        min[0] = Math.min(min[0], x);
        min[1] = Math.min(min[1], y);
        min[2] = Math.min(min[2], z);

        max[0] = Math.max(max[0], x);
        max[1] = Math.max(max[1], y);
        max[2] = Math.max(max[2], z);
    }

    const voxels = new Map<string, number[]>();

    // --- triangle loop ---
    for (let i = 0; i < obj.indicesCount; i += 3) {
        const i0 = indices[i] * vStride;
        const i1 = indices[i + 1] * vStride;
        const i2 = indices[i + 2] * vStride;

        const v0: Vec3 = [vertices[i0], vertices[i0 + 1], vertices[i0 + 2]];
        const v1: Vec3 = [vertices[i1], vertices[i1 + 1], vertices[i1 + 2]];
        const v2: Vec3 = [vertices[i2], vertices[i2 + 1], vertices[i2 + 2]];

        const c0: Vec3 = [vertices[i0 + 4], vertices[i0 + 5], vertices[i0 + 6]];
        const c1: Vec3 = [vertices[i1 + 4], vertices[i1 + 5], vertices[i1 + 6]];
        const c2: Vec3 = [vertices[i2 + 4], vertices[i2 + 5], vertices[i2 + 6]];

        const triMin: Vec3 = [
            Math.min(v0[0], v1[0], v2[0]),
            Math.min(v0[1], v1[1], v2[1]),
            Math.min(v0[2], v1[2], v2[2]),
        ];

        const triMax: Vec3 = [
            Math.max(v0[0], v1[0], v2[0]),
            Math.max(v0[1], v1[1], v2[1]),
            Math.max(v0[2], v1[2], v2[2]),
        ];

        const ix0 = Math.floor((triMin[0] - min[0]) / voxelSize);
        const iy0 = Math.floor((triMin[1] - min[1]) / voxelSize);
        const iz0 = Math.floor((triMin[2] - min[2]) / voxelSize);

        const ix1 = Math.ceil((triMax[0] - min[0]) / voxelSize);
        const iy1 = Math.ceil((triMax[1] - min[1]) / voxelSize);
        const iz1 = Math.ceil((triMax[2] - min[2]) / voxelSize);

        for (let z = iz0; z <= iz1; z++) {
            for (let y = iy0; y <= iy1; y++) {
                for (let x = ix0; x <= ix1; x++) {
                    const cx = min[0] + (x + 0.5) * voxelSize;
                    const cy = min[1] + (y + 0.5) * voxelSize;
                    const cz = min[2] + (z + 0.5) * voxelSize;

                    const half = voxelSize * 0.5;

                    if (!triangleBoxIntersect(v0, v1, v2, [cx, cy, cz], half))
                        continue;

                    const key = `${x},${y},${z}`;

                    if (!voxels.has(key)) {
                        voxels.set(key, [
                            cx,
                            cy,
                            cz,
                            (c0[0] + c1[0] + c2[0]) / 3,
                            (c0[1] + c1[1] + c2[1]) / 3,
                            (c0[2] + c1[2] + c2[2]) / 3,
                        ]);
                    }
                }
            }
        }
    }

    // --- output buffer ---
    const out = new Float32Array(voxels.size * stride);
    let o = 0;

    for (const v of voxels.values()) {
        out[o++] = v[0];
        out[o++] = v[1];
        out[o++] = v[2];
        out[o++] = 0.0;
        out[o++] = v[3];
        out[o++] = v[4];
        out[o++] = v[5];
        out[o++] = 0.0;
    }

    return out;
}

function triangleBoxIntersect(
    v0: Vec3,
    v1: Vec3,
    v2: Vec3,
    c: Vec3,
    h: number,
): boolean {
    // Move triangle into box space
    const tv0 = sub(v0, c);
    const tv1 = sub(v1, c);
    const tv2 = sub(v2, c);

    const e0 = sub(tv1, tv0);
    const e1 = sub(tv2, tv1);
    const e2 = sub(tv0, tv2);

    const axes = [
        cross(e0, [1, 0, 0]),
        cross(e0, [0, 1, 0]),
        cross(e0, [0, 0, 1]),
        cross(e1, [1, 0, 0]),
        cross(e1, [0, 1, 0]),
        cross(e1, [0, 0, 1]),
        cross(e2, [1, 0, 0]),
        cross(e2, [0, 1, 0]),
        cross(e2, [0, 0, 1]),
    ];

    for (const a of axes) {
        if (!axisOverlap(a, tv0, tv1, tv2, h)) return false;
    }

    for (let i = 0; i < 3; i++) {
        const min = Math.min(tv0[i], tv1[i], tv2[i]);
        const max = Math.max(tv0[i], tv1[i], tv2[i]);
        if (min > h || max < -h) return false;
    }

    const n = cross(e0, e1);
    if (!axisOverlap(n, tv0, tv1, tv2, h)) return false;

    return true;
}

function sub(a: Vec3, b: Vec3): Vec3 {
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function cross(a: Vec3, b: Vec3): Vec3 {
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ];
}

function dot(a: Vec3, b: Vec3): number {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function axisOverlap(
    axis: Vec3,
    v0: Vec3,
    v1: Vec3,
    v2: Vec3,
    h: number,
): boolean {
    const p0 = dot(v0, axis);
    const p1 = dot(v1, axis);
    const p2 = dot(v2, axis);

    const r = h * (Math.abs(axis[0]) + Math.abs(axis[1]) + Math.abs(axis[2]));

    return Math.max(-Math.max(p0, p1, p2), Math.min(p0, p1, p2)) <= r;
}

//////////// SHADER ////////////

const directory: string = "./shaders/softwarerasterizer/";
const includesDirectory: string = "./shaders/softwarerasterizer/includes/";
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

//////////// STATS ////////////

export function toDelta(
    timingsNanoseconds: BigInt64Array,
    a: int,
    b: int,
): float {
    return Number(timingsNanoseconds[b] - timingsNanoseconds[a]) / 1_000_000;
}
