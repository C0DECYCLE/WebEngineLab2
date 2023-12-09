/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, October 2023
 */

import { int, float, Nullable } from "../types/utilities/utils.type.js";
import { clear } from "./utilities/utils.js";

type OBJVertex = [float, float, float];
export type OBJParseResult = {
    positions: Float32Array;
    positionsCount: int;
    indices?: Uint32Array;
    indicesCount?: int;
};

export class OBJParser {
    public static readonly Standard: OBJParser = new OBJParser();

    private readonly vertexCache: OBJVertex[];
    private readonly positionCache: float[];
    private readonly indexCache: int[];

    public constructor() {
        this.vertexCache = [];
        this.positionCache = [];
        this.indexCache = [];
    }

    public parse(raw: string, indexed: boolean = false): OBJParseResult {
        this.reset();
        const regExp: RegExp = /(\w*)(?: )*(.*)/;
        const lines: string[] = raw.split("\n");
        for (let i: int = 0; i < lines.length; ++i) {
            this.parseLine(regExp, lines[i].trim(), indexed);
        }
        const result: OBJParseResult = {
            positions: new Float32Array(this.positionCache),
            positionsCount: this.positionCache.length / 4,
        } as OBJParseResult;
        if (indexed) {
            result.indices = new Uint32Array(this.indexCache);
            result.indicesCount = this.indexCache.length;
        }
        this.reset();
        return result;
    }

    private parseLine(regExp: RegExp, line: string, indexed: boolean): void {
        const m: Nullable<RegExpExecArray> = regExp.exec(line);
        if (line === "" || line.startsWith("#") || !m) {
            return;
        }
        const parts: string[] = line.split(/\s+/).slice(1);
        switch (m[1]) {
            case "v":
                return this.keywordV(parts, indexed);
            case "f":
                return this.keywordF(parts, indexed);
        }
    }

    private keywordV(parts: string[], indexed: boolean): void {
        if (parts.length < 3) {
            throw new Error(`ObjParser: Obj file missing vertex part.`);
        }
        const x: float = parseFloat(parts[0]);
        const y: float = parseFloat(parts[1]);
        const z: float = parseFloat(parts[2]);
        this.vertexCache.push([x, y, z]);
        if (!indexed) {
            return;
        }
        this.positionCache.push(x, y, z, 0);
    }

    private keywordF(parts: string[], indexed: boolean): void {
        const a: int = parseInt(parts[0]) - 1;
        const b: int = parseInt(parts[1]) - 1;
        const c: int = parseInt(parts[2]) - 1;
        if (!indexed) {
            this.positionCache.push(...this.vertexCache[a].slice(0, 3), 0);
            this.positionCache.push(...this.vertexCache[b].slice(0, 3), 0);
            this.positionCache.push(...this.vertexCache[c].slice(0, 3), 0);
            return;
        }
        this.indexCache.push(a, b, c);
    }

    public reset(): void {
        clear(this.vertexCache);
        clear(this.positionCache);
        clear(this.indexCache);
    }

    public destroy(): void {
        this.reset();
    }
}
