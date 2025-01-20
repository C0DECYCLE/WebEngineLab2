/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, October 2023
 */

import { Nullable } from "../types/utilities/utils.type.js";
import { OBJParseResult, OBJParser } from "../src/OBJParser.js";

describe("OBJParser", () => {
    const raw: string =
        "# Blender 3.4.1\n# www.blender.org\no Cube\nv -0.500000 -0.500000 0.500000\nv -0.500000 0.500000 0.500000\nv -0.500000 -0.500000 -0.500000\nv -0.500000 0.500000 -0.500000\nv 0.500000 -0.500000 0.500000\nv 0.500000 0.500000 0.500000\nv 0.500000 -0.500000 -0.500000\nv 0.500000 0.500000 -0.500000\ns 0\nf 2 3 1\nf 4 7 3\nf 8 5 7\nf 6 1 5\nf 7 1 3\nf 4 6 8\nf 2 4 3\nf 4 8 7\nf 8 6 5\nf 6 2 1\nf 7 5 1\nf 4 2 6";
    const parsedVerticesUnindexed: string =
        "-0.5,0.5,0.5,0,-0.5,-0.5,-0.5,0,-0.5,-0.5,0.5,0,-0.5,0.5,-0.5,0,0.5,-0.5,-0.5,0,-0.5,-0.5,-0.5,0,0.5,0.5,-0.5,0,0.5,-0.5,0.5,0,0.5,-0.5,-0.5,0,0.5,0.5,0.5,0,-0.5,-0.5,0.5,0,0.5,-0.5,0.5,0,0.5,-0.5,-0.5,0,-0.5,-0.5,0.5,0,-0.5,-0.5,-0.5,0,-0.5,0.5,-0.5,0,0.5,0.5,0.5,0,0.5,0.5,-0.5,0,-0.5,0.5,0.5,0,-0.5,0.5,-0.5,0,-0.5,-0.5,-0.5,0,-0.5,0.5,-0.5,0,0.5,0.5,-0.5,0,0.5,-0.5,-0.5,0,0.5,0.5,-0.5,0,0.5,0.5,0.5,0,0.5,-0.5,0.5,0,0.5,0.5,0.5,0,-0.5,0.5,0.5,0,-0.5,-0.5,0.5,0,0.5,-0.5,-0.5,0,0.5,-0.5,0.5,0,-0.5,-0.5,0.5,0,-0.5,0.5,-0.5,0,-0.5,0.5,0.5,0,0.5,0.5,0.5,0";
    const parsedVerticesIndexed: string =
        "-0.5,-0.5,0.5,0,-0.5,0.5,0.5,0,-0.5,-0.5,-0.5,0,-0.5,0.5,-0.5,0,0.5,-0.5,0.5,0,0.5,0.5,0.5,0,0.5,-0.5,-0.5,0,0.5,0.5,-0.5,0";
    const parsedIndices: string =
        "1,2,0,3,6,2,7,4,6,5,0,4,6,0,2,3,5,7,1,3,2,3,7,6,7,5,4,5,1,0,6,4,0,3,1,5";
    let parser: Nullable<OBJParser> = null;

    beforeEach(() => {
        parser = new OBJParser();
    });

    afterEach(() => {
        parser?.destroy();
        parser = null;
    });

    test("parse", () => {
        expect(parser).not.toBeNull();
        const result: OBJParseResult = parser!.parse(raw);
        const vertices: Float32Array = result.vertices;
        expect(vertices).toBeInstanceOf(Float32Array);
        expect(vertices.toString()).toBe(parsedVerticesUnindexed);
    });

    test("parse indexed", () => {
        expect(parser).not.toBeNull();
        const result: OBJParseResult = parser!.parse(raw, true);
        const vertices: Float32Array = result.vertices;
        expect(vertices).toBeInstanceOf(Float32Array);
        expect(vertices.toString()).toBe(parsedVerticesIndexed);
        const indices: Uint32Array = result.indices!;
        expect(indices).toBeInstanceOf(Uint32Array);
        expect(indices.toString()).toBe(parsedIndices);
    });
});
