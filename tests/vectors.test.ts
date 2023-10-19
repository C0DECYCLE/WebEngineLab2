/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, October 2023
 */

import { Vector2, Vector3 } from "../src/utilities/vectors";

describe("Vectors", () => {
    test("Vector2 Addition", () => {
        const a: Vector2 = new Vector2(1, 2);
        const b: Vector2 = new Vector2(2, 1);
        const expected: Vector2 = new Vector2(3, 3);
        expect(a.add(b)).toEqual(expected);
    });

    test("Vector3 Addition", () => {
        const a: Vector3 = new Vector3(1, 2, 3);
        const b: Vector3 = new Vector3(3, 2, 1);
        const expected: Vector2 = new Vector3(4, 4, 4);
        expect(a.add(b)).toEqual(expected);
    });
});
