/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger
 */

import { Mesh } from "./clusterize";

//import { log } from "../utilities/logger.js";

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
