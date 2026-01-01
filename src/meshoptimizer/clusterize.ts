/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger
 */

//import { log } from "../utilities/logger.js";

export type Mesh = {
    positions: Float32Array;
    indices: Uint32Array;
};

const maxTris: number = 128;
const maxVerts: number = 384;
const minTris: number = 64;

function buildMeshletsMinVertexSharing(indices: Uint32Array): number[][] {
    const triCount = indices.length / 3;

    const triVerts: number[][] = new Array(triCount);
    for (let t = 0; t < triCount; t++) {
        triVerts[t] = [
            indices[t * 3 + 0],
            indices[t * 3 + 1],
            indices[t * 3 + 2],
        ];
    }

    // --- adjacency ---
    const adjacency: number[][] = Array.from({ length: triCount }, () => []);
    const edgeMap = new Map<string, number>();

    for (let t = 0; t < triCount; t++) {
        const v = triVerts[t];
        const edges: [number, number][] = [
            [v[0], v[1]],
            [v[1], v[2]],
            [v[2], v[0]],
        ];
        for (const [a, b] of edges) {
            const key = a < b ? `${a},${b}` : `${b},${a}`;
            const other = edgeMap.get(key);
            if (other !== undefined) {
                adjacency[t].push(other);
                adjacency[other].push(t);
            } else {
                edgeMap.set(key, t);
            }
        }
    }

    const claimed = new Uint8Array(triCount);
    const meshlets: number[][] = [];

    for (let seed = 0; seed < triCount; seed++) {
        if (claimed[seed]) continue;

        const localUsed = new Uint8Array(triCount);
        const meshlet: number[] = [];
        const vertSet = new Set<number>();

        // seed
        localUsed[seed] = 1;
        meshlet.push(seed);
        triVerts[seed].forEach((v) => vertSet.add(v));

        const frontier = new Set<number>(adjacency[seed]);

        while (meshlet.length < maxTris && frontier.size > 0) {
            let bestTri = -1;
            let bestScore = -Infinity;

            for (const t of frontier) {
                if (claimed[t] || localUsed[t]) continue;

                let shared = 0;
                let added = 0;
                for (const v of triVerts[t]) {
                    if (vertSet.has(v)) shared++;
                    else added++;
                }

                if (vertSet.size + added > maxVerts) continue;

                // Allow expansion even with low score if meshlet is small
                const score =
                    meshlet.length < minTris
                        ? shared * 5 - added * 5
                        : shared * 10 - added * 20;

                if (score > bestScore) {
                    bestScore = score;
                    bestTri = t;
                }
            }

            if (bestTri === -1) break;

            localUsed[bestTri] = 1;
            meshlet.push(bestTri);
            triVerts[bestTri].forEach((v) => vertSet.add(v));

            frontier.delete(bestTri);
            adjacency[bestTri].forEach((n) => frontier.add(n));
        }

        // Commit meshlet
        for (const t of meshlet) claimed[t] = 1;
        meshlets.push(meshlet);
    }

    return meshlets;
}

function mergeSmallMeshlets(
    meshlets: number[][],
    indices: Uint32Array,
): number[][] {
    const triVerts: number[][] = [];
    for (let t = 0; t < indices.length / 3; t++) {
        triVerts[t] = [
            indices[t * 3 + 0],
            indices[t * 3 + 1],
            indices[t * 3 + 2],
        ];
    }

    // Build triangle -> meshlet lookup
    const triToMeshlet = new Int32Array(indices.length / 3).fill(-1);
    meshlets.forEach((m, i) => {
        for (const t of m) triToMeshlet[t] = i;
    });

    // Precompute meshlet vertex sets
    const meshletVerts: Set<number>[] = meshlets.map(() => new Set());
    meshlets.forEach((m, i) => {
        for (const t of m) {
            for (const v of triVerts[t]) {
                meshletVerts[i].add(v);
            }
        }
    });

    const alive = new Uint8Array(meshlets.length).fill(1);

    for (let i = 0; i < meshlets.length; i++) {
        if (!alive[i]) continue;
        if (meshlets[i].length >= minTris) continue;

        let bestTarget = -1;
        let bestAddedVerts = Infinity;

        // Find neighboring meshlets
        const neighborSet = new Set<number>();
        for (const t of meshlets[i]) {
            for (const v of triVerts[t]) {
                // find triangles sharing this vertex
                for (let ot = 0; ot < triToMeshlet.length; ot++) {
                    if (triToMeshlet[ot] === i) continue;
                }
            }
        }

        // Easier: brute-force neighbors (small meshlets are rare)
        for (let j = 0; j < meshlets.length; j++) {
            if (i === j || !alive[j]) continue;

            const triCount = meshlets[i].length + meshlets[j].length;
            if (triCount > maxTris) continue;

            // Count added vertices
            let addedVerts = 0;
            for (const v of meshletVerts[i]) {
                if (!meshletVerts[j].has(v)) addedVerts++;
            }

            if (meshletVerts[j].size + addedVerts > maxVerts) continue;

            if (addedVerts < bestAddedVerts) {
                bestAddedVerts = addedVerts;
                bestTarget = j;
            }
        }

        if (bestTarget !== -1) {
            // Merge i into bestTarget
            for (const t of meshlets[i]) {
                meshlets[bestTarget].push(t);
                triToMeshlet[t] = bestTarget;
            }

            for (const v of meshletVerts[i]) {
                meshletVerts[bestTarget].add(v);
            }

            alive[i] = 0;
        }
    }

    // Compact meshlet list
    const merged: number[][] = [];
    for (let i = 0; i < meshlets.length; i++) {
        if (alive[i]) merged.push(meshlets[i]);
    }

    return merged;
}

function buildMeshlet(mesh: Mesh, triangleIndices: number[]): Mesh {
    const vertMap = new Map<number, number>();
    const verts: number[] = [];
    const inds: number[] = [];

    for (const tri of triangleIndices) {
        for (let k = 0; k < 3; k++) {
            const v = mesh.indices[tri * 3 + k];
            let local = vertMap.get(v);
            if (local === undefined) {
                local = verts.length / 4;
                vertMap.set(v, local);
                verts.push(
                    mesh.positions[v * 4 + 0],
                    mesh.positions[v * 4 + 1],
                    mesh.positions[v * 4 + 2],
                    mesh.positions[v * 4 + 3],
                );
            }
            inds.push(local);
        }
    }

    return {
        positions: new Float32Array(verts),
        indices: new Uint32Array(inds),
    };
}

export function clusterizeTriangles(mesh: Mesh): Mesh[] {
    let clusters: number[][] = buildMeshletsMinVertexSharing(mesh.indices);
    clusters = mergeSmallMeshlets(clusters, mesh.indices);
    const meshlets: Mesh[] = clusters.map((cluster: number[]) =>
        buildMeshlet(mesh, cluster),
    );
    return meshlets;
}
