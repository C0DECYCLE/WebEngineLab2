/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, October 2023
 */

import { int, float, Nullable } from "../types/utilities/utils.type.js";
import { clear } from "./utilities/utils.js";

type OBJPolygon = [float, float, float];

export class OBJParser {
    private readonly vertices: float[] = [];
    private readonly polygons: OBJPolygon[] = [];

    public parse(raw: string): Float32Array {
        this.reset();
        this.polygons.push([0.0, 0.0, 0.0]);
        const regExp: RegExp = /(\w*)(?: )*(.*)/;
        const lines: string[] = raw.split("\n");
        for (let i: int = 0; i < lines.length; ++i) {
            this.parseLine(regExp, lines[i].trim());
        }
        const result: Float32Array = new Float32Array(this.vertices);
        this.reset();
        return result;
    }

    public reset(): void {
        clear(this.vertices);
        clear(this.polygons);
    }

    public destroy(): void {
        this.reset();
    }

    private parseLine(regExp: RegExp, line: string): void {
        const m: Nullable<RegExpExecArray> = regExp.exec(line);
        if (line === "" || line.startsWith("#") || !m) {
            return;
        }
        const [, keyword, _unparsedArgs] = m;
        const parts: string[] = line.split(/\s+/).slice(1);
        switch (keyword) {
            case "v":
                return this.keywordV(parts);
            case "f":
                return this.keywordF(parts);
            default:
                return;
        }
    }

    private keywordV(parts: string[]): void {
        if (parts.length < 3) {
            throw new Error(`ObjParser: Obj file missing vertex part.`);
        }
        this.polygons.push([
            parseFloat(parts[0]),
            parseFloat(parts[1]),
            parseFloat(parts[2]),
        ]);
    }

    private keywordF(parts: string[]): void {
        const a: int = parseInt(parts[0]);
        const b: int = parseInt(parts[1]);
        const c: int = parseInt(parts[2]);
        this.vertices.push(...this.polygons[a].slice(0, 3), 0.0);
        this.vertices.push(...this.polygons[b].slice(0, 3), 0.0);
        this.vertices.push(...this.polygons[c].slice(0, 3), 0.0);
    }
}
