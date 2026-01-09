/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger
 */

import { METIS_OPTION, partitionGraph } from "./METISb/partitionGraph.js";

export type Mesh = {
    positions: Float32Array;
    indices: Uint32Array;
};

type WeightedAdjacency = {
    xadj: number[];
    adjncy: number[];
    adjwgt: number[];
    triangleAdjacency: number[][];
};

function buildWeightedTriangleAdjacency(mesh: Mesh): WeightedAdjacency {
    const indices = mesh.indices;
    const triCount = indices.length / 3;

    const edgeMap = new Map<string, number[]>();

    function edgeKey(a: number, b: number) {
        return a < b ? `${a}_${b}` : `${b}_${a}`;
    }

    for (let t = 0; t < triCount; t++) {
        const i0 = indices[t * 3 + 0];
        const i1 = indices[t * 3 + 1];
        const i2 = indices[t * 3 + 2];

        edgeMap.get(edgeKey(i0, i1))?.push(t) ??
            edgeMap.set(edgeKey(i0, i1), [t]);
        edgeMap.get(edgeKey(i1, i2))?.push(t) ??
            edgeMap.set(edgeKey(i1, i2), [t]);
        edgeMap.get(edgeKey(i2, i0))?.push(t) ??
            edgeMap.set(edgeKey(i2, i0), [t]);
    }

    const triangleAdjacency: number[][] = Array.from(
        { length: triCount },
        () => [],
    );
    const weights: number[][] = Array.from({ length: triCount }, () => []);

    for (const tris of edgeMap.values()) {
        if (tris.length === 2) {
            const [a, b] = tris;
            triangleAdjacency[a].push(b);
            triangleAdjacency[b].push(a);
            weights[a].push(2);
            weights[b].push(2);
        }
    }

    const xadj: number[] = [0];
    const adjncy: number[] = [];
    const adjwgt: number[] = [];

    for (let i = 0; i < triCount; i++) {
        adjncy.push(...triangleAdjacency[i]);
        adjwgt.push(...weights[i]);
        xadj.push(adjncy.length);
    }

    return { xadj, adjncy, adjwgt, triangleAdjacency };
}

function buildMeshletsFromClusters(mesh: Mesh, clusters: number[][]): Mesh[] {
    const VERTEX_STRIDE = 4;
    const srcPos = mesh.positions;
    const srcIdx = mesh.indices;

    const meshlets: Mesh[] = [];

    for (const tris of clusters) {
        const indexMap = new Map<number, number>();
        const positions: number[] = [];
        const indices: number[] = [];

        function remap(v: number): number {
            let idx = indexMap.get(v);
            if (idx === undefined) {
                idx = indexMap.size;
                indexMap.set(v, idx);

                const srcOffset = v * VERTEX_STRIDE;
                const dstOffset = idx * VERTEX_STRIDE;

                positions[dstOffset + 0] = srcPos[srcOffset + 0];
                positions[dstOffset + 1] = srcPos[srcOffset + 1];
                positions[dstOffset + 2] = srcPos[srcOffset + 2];
                positions[dstOffset + 3] = srcPos[srcOffset + 3];
            }
            return idx;
        }

        for (const t of tris) {
            const i0 = srcIdx[t * 3 + 0];
            const i1 = srcIdx[t * 3 + 1];
            const i2 = srcIdx[t * 3 + 2];

            indices.push(remap(i0), remap(i1), remap(i2));
        }

        meshlets.push({
            positions: new Float32Array(positions),
            indices: new Uint32Array(indices),
        });
    }

    return meshlets;
}

export type TriangleClusteringResult = {
    meshlets: Mesh[];
    clusters: number[][]; // triangle indices per meshlet
    triangleAdjacency: number[][]; // triangle → triangle adjacency
};

export async function clusterizeTriangles(
    mesh: Mesh,
): Promise<TriangleClusteringResult> {
    const triCount = mesh.indices.length / 3;
    const nparts = Math.ceil(triCount / 126);

    const { xadj, adjncy, adjwgt, triangleAdjacency } =
        buildWeightedTriangleAdjacency(mesh);

    const clusters = await partitionGraph(xadj, adjncy, adjwgt, nparts, {
        //[METIS_OPTION.NUMBERING]: 0,
        //[METIS_OPTION.CONTIG]: 1,
        [METIS_OPTION.UFACTOR]: 1,
    });

    const meshlets = buildMeshletsFromClusters(mesh, clusters);

    return {
        meshlets,
        clusters, // <-- THIS is clusterTriangles
        triangleAdjacency, // <-- THIS is triangleAdjacency
    };
}
