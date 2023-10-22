/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, October 2023
 */

import { Nullable } from "../types/utilities/utils.type.js";
import { OBJParser } from "../src/OBJParser.js";

describe("OBJParser", () => {
    const raw: string =
        "# Blender 3.4.1\n# www.blender.org\no Cube\nv -0.500000 -0.500000 0.500000\nv -0.500000 0.500000 0.500000\nv -0.500000 -0.500000 -0.500000\nv -0.500000 0.500000 -0.500000\nv 0.500000 -0.500000 0.500000\nv 0.500000 0.500000 0.500000\nv 0.500000 -0.500000 -0.500000\nv 0.500000 0.500000 -0.500000\ns 0\nf 2 3 1\nf 4 7 3\nf 8 5 7\nf 6 1 5\nf 7 1 3\nf 4 6 8\nf 2 4 3\nf 4 8 7\nf 8 6 5\nf 6 2 1\nf 7 5 1\nf 4 2 6";
    const parsed: string =
        "-0.5,0.5,0.5,0,-0.5,-0.5,-0.5,0,-0.5,-0.5,0.5,0,-0.5,0.5,-0.5,0,0.5,-0.5,-0.5,0,-0.5,-0.5,-0.5,0,0.5,0.5,-0.5,0,0.5,-0.5,0.5,0,0.5,-0.5,-0.5,0,0.5,0.5,0.5,0,-0.5,-0.5,0.5,0,0.5,-0.5,0.5,0,0.5,-0.5,-0.5,0,-0.5,-0.5,0.5,0,-0.5,-0.5,-0.5,0,-0.5,0.5,-0.5,0,0.5,0.5,0.5,0,0.5,0.5,-0.5,0,-0.5,0.5,0.5,0,-0.5,0.5,-0.5,0,-0.5,-0.5,-0.5,0,-0.5,0.5,-0.5,0,0.5,0.5,-0.5,0,0.5,-0.5,-0.5,0,0.5,0.5,-0.5,0,0.5,0.5,0.5,0,0.5,-0.5,0.5,0,0.5,0.5,0.5,0,-0.5,0.5,0.5,0,-0.5,-0.5,0.5,0,0.5,-0.5,-0.5,0,0.5,-0.5,0.5,0,-0.5,-0.5,0.5,0,-0.5,0.5,-0.5,0,-0.5,0.5,0.5,0,0.5,0.5,0.5,0";
    let parser: Nullable<OBJParser> = null;

    beforeEach(() => {
        parser = new OBJParser();
    });

    afterEach(() => {
        parser?.destroy();
        parser = null;
    });

    test("parse", async () => {
        expect(parser).not.toBeNull();
        const result: Float32Array = parser!.parse(raw);
        expect(result).toBeInstanceOf(Float32Array);
        expect(result.toString()).toBe(parsed);
    });
});
