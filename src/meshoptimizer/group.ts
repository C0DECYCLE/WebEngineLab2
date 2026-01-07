/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger
 */

import { Mesh } from "./clusterize";
import { METIS_OPTION, partitionGraph } from "./METISb/partitionGraph";

//import { log } from "../utilities/logger.js";

/*
interface MeshletGroup {
    meshletIndices: number[]; // indices into original meshlets array
    positions: Float32Array; // concatenated positions of all meshlets in group
}

const maxMeshletsPerGroup: number = 8;
const maxVertsPerGroup: number = 256;

export function buildMeshletLODGroupsFromPositions(
    meshlets: Mesh[],
): MeshletGroup[] {
    const assigned = new Uint8Array(meshlets.length);
    const groups: MeshletGroup[] = [];

    for (let seed = 0; seed < meshlets.length; seed++) {
        if (assigned[seed]) continue;

        const groupMeshlets: number[] = [];
        const vertSet = new Map<string, number>(); // use "x,y,z,w" key to dedupe
        const posList: number[] = [];

        // Seed meshlet
        groupMeshlets.push(seed);
        assigned[seed] = 1;

        const seedPositions = meshlets[seed].positions;
        const vertCount = seedPositions.length / 4;
        for (let i = 0; i < vertCount; i++) {
            const key = `${seedPositions[i * 4 + 0]},${
                seedPositions[i * 4 + 1]
            },${seedPositions[i * 4 + 2]},${seedPositions[i * 4 + 3]}`;
            if (!vertSet.has(key)) {
                vertSet.set(key, posList.length / 4);
                posList.push(
                    seedPositions[i * 4 + 0],
                    seedPositions[i * 4 + 1],
                    seedPositions[i * 4 + 2],
                    seedPositions[i * 4 + 3],
                );
            }
        }

        // Candidate meshlets
        const frontier = new Set<number>();
        for (let i = 0; i < meshlets.length; i++) {
            if (!assigned[i] && i !== seed) frontier.add(i);
        }

        while (
            groupMeshlets.length < maxMeshletsPerGroup &&
            frontier.size > 0
        ) {
            let bestMeshlet = -1;
            let bestScore = -Infinity;

            for (const idx of frontier) {
                const m = meshlets[idx];
                let shared = 0;
                let added = 0;

                const mVertCount = m.positions.length / 4;
                for (let vi = 0; vi < mVertCount; vi++) {
                    const key = `${m.positions[vi * 4 + 0]},${
                        m.positions[vi * 4 + 1]
                    },${m.positions[vi * 4 + 2]},${m.positions[vi * 4 + 3]}`;
                    if (vertSet.has(key)) shared++;
                    else added++;
                }

                if (vertSet.size + added > maxVertsPerGroup) continue;

                const score = shared * 10 - added * 20;
                if (score > bestScore) {
                    bestScore = score;
                    bestMeshlet = idx;
                }
            }

            if (bestMeshlet === -1) break;

            // Accept meshlet
            groupMeshlets.push(bestMeshlet);
            assigned[bestMeshlet] = 1;

            const newMeshlet = meshlets[bestMeshlet];
            const newVertCount = newMeshlet.positions.length / 4;
            for (let vi = 0; vi < newVertCount; vi++) {
                const key = `${newMeshlet.positions[vi * 4 + 0]},${
                    newMeshlet.positions[vi * 4 + 1]
                },${newMeshlet.positions[vi * 4 + 2]},${
                    newMeshlet.positions[vi * 4 + 3]
                }`;
                if (!vertSet.has(key)) {
                    vertSet.set(key, posList.length / 4);
                    posList.push(
                        newMeshlet.positions[vi * 4 + 0],
                        newMeshlet.positions[vi * 4 + 1],
                        newMeshlet.positions[vi * 4 + 2],
                        newMeshlet.positions[vi * 4 + 3],
                    );
                }
            }

            frontier.delete(bestMeshlet);
        }

        groups.push({
            meshletIndices: groupMeshlets,
            positions: new Float32Array(posList),
        });
    }

    return groups;
}

export function groupClusters(meshlets: Mesh[]): MeshletGroup[] {
    const groups = buildMeshletLODGroupsFromPositions(meshlets);
    return groups;
}
*/

function buildTriangleToClusterMap(clusters: number[][]): Int32Array {
    let maxTri = 0;
    for (const c of clusters) {
        for (const t of c) maxTri = Math.max(maxTri, t);
    }

    const triToCluster = new Int32Array(maxTri + 1);
    triToCluster.fill(-1);

    clusters.forEach((tris, ci) => {
        for (const t of tris) triToCluster[t] = ci;
    });

    return triToCluster;
}

type ClusterEdgeKey = string;

function clusterEdgeKey(a: number, b: number): ClusterEdgeKey {
    return a < b ? `${a}_${b}` : `${b}_${a}`;
}

function buildClusterAdjacencyFromTopology(
    triangleAdjacency: number[][],
    clusters: number[][],
): Map<ClusterEdgeKey, number> {
    const triToCluster = buildTriangleToClusterMap(clusters);
    const edgeWeights = new Map<ClusterEdgeKey, number>();

    for (let t = 0; t < triangleAdjacency.length; t++) {
        const c0 = triToCluster[t];
        if (c0 < 0) continue;

        for (const t2 of triangleAdjacency[t]) {
            const c1 = triToCluster[t2];
            if (c1 < 0 || c0 === c1) continue;

            const key = clusterEdgeKey(c0, c1);
            edgeWeights.set(key, (edgeWeights.get(key) ?? 0) + 1);
        }
    }

    return edgeWeights;
}

function computeMeshletCentroids(meshlets: Mesh[]): Float32Array {
    const centroids = new Float32Array(meshlets.length * 3);

    meshlets.forEach((m, i) => {
        let x = 0,
            y = 0,
            z = 0;
        const stride = 4;
        const count = m.positions.length / stride;

        for (let v = 0; v < count; v++) {
            x += m.positions[v * stride + 0];
            y += m.positions[v * stride + 1];
            z += m.positions[v * stride + 2];
        }

        centroids[i * 3 + 0] = x / count;
        centroids[i * 3 + 1] = y / count;
        centroids[i * 3 + 2] = z / count;
    });

    return centroids;
}

function addSpatialAdjacency(
    centroids: Float32Array,
    edgeWeights: Map<ClusterEdgeKey, number>,
    k = 3,
    spatialWeight = 1,
) {
    const n = centroids.length / 3;

    for (let i = 0; i < n; i++) {
        const cx = centroids[i * 3 + 0];
        const cy = centroids[i * 3 + 1];
        const cz = centroids[i * 3 + 2];

        const dists: { j: number; d: number }[] = [];

        for (let j = 0; j < n; j++) {
            if (i === j) continue;
            const dx = centroids[j * 3 + 0] - cx;
            const dy = centroids[j * 3 + 1] - cy;
            const dz = centroids[j * 3 + 2] - cz;
            dists.push({ j, d: dx * dx + dy * dy + dz * dz });
        }

        dists.sort((a, b) => a.d - b.d);

        for (let t = 0; t < k; t++) {
            const j = dists[t].j;
            const key = clusterEdgeKey(i, j);
            edgeWeights.set(key, (edgeWeights.get(key) ?? 0) + spatialWeight);
        }
    }
}

function buildMetisGraphFromClusterEdges(
    clusterCount: number,
    edgeWeights: Map<ClusterEdgeKey, number>,
) {
    const adjacency: number[][] = Array.from(
        { length: clusterCount },
        () => [],
    );
    const weights: number[][] = Array.from({ length: clusterCount }, () => []);

    for (const [key, w] of edgeWeights) {
        const [a, b] = key.split("_").map(Number);
        adjacency[a].push(b);
        weights[a].push(w);
        adjacency[b].push(a);
        weights[b].push(w);
    }

    const xadj: number[] = [0];
    const adjncy: number[] = [];
    const adjwgt: number[] = [];

    for (let i = 0; i < clusterCount; i++) {
        adjncy.push(...adjacency[i]);
        adjwgt.push(...weights[i]);
        xadj.push(adjncy.length);
    }

    return { xadj, adjncy, adjwgt };
}

export async function groupClusters(
    meshlets: Mesh[],
    clusterTriangles: number[][],
    triangleAdjacency: number[][],
): Promise<number[][]> {
    const clusterCount = meshlets.length;
    const nparts = Math.ceil(clusterCount / 8);

    const topoEdges = buildClusterAdjacencyFromTopology(
        triangleAdjacency,
        clusterTriangles,
    );

    const centroids = computeMeshletCentroids(meshlets);
    addSpatialAdjacency(centroids, topoEdges);

    const { xadj, adjncy, adjwgt } = buildMetisGraphFromClusterEdges(
        clusterCount,
        topoEdges,
    );

    const groups = await partitionGraph(xadj, adjncy, adjwgt, nparts, {
        [METIS_OPTION.NUMBERING]: 0,
        [METIS_OPTION.CONTIG]: 1,
        [METIS_OPTION.UFACTOR]: 1,
    });

    return groups;
}
