/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, October 2023
 */

import { Vec3 } from "../../src/utilities/Vec3.js";
import { Mat4 } from "../../src/utilities/Mat4.js";
import {
    expectFloatArrayToBeCloseTo,
    expectVec3ToBeCloseTo,
} from "../utils.js";

describe("Vec3", () => {
    test("get", () => {
        expectVec3ToBeCloseTo(new Vec3(32, 51.2, -0.89), 32, 51.2, -0.89);
    });

    test("set", () => {
        const a: Vec3 = new Vec3();
        expectVec3ToBeCloseTo(a, 0, 0, 0);
        a.x = -3.2;
        a.y = 89234;
        a.z = 0.012;
        expectVec3ToBeCloseTo(a, -3.2, 89234, 0.012);
        expectVec3ToBeCloseTo(a.set(4, 5, 6), 4, 5, 6);
    });

    test("add", () => {
        const a: Vec3 = new Vec3(1, 2, 3);
        const b: Vec3 = new Vec3(4, 5, 6);
        expectVec3ToBeCloseTo(a.add(b), 5, 7, 9);
        expectVec3ToBeCloseTo(b.add(3, 2, 1), 7, 7, 7);
        expectVec3ToBeCloseTo(a.add(0, 0, 0), 5, 7, 9);
        expectVec3ToBeCloseTo(b.add(2), 9, 9, 9);
    });

    test("sub", () => {
        const a: Vec3 = new Vec3(4, 5, 6);
        const b: Vec3 = new Vec3(1, 2, 3);
        expectVec3ToBeCloseTo(a.sub(b), 3, 3, 3);
        expectVec3ToBeCloseTo(b.sub(3, 2, 1), -2, 0, 2);
        expectVec3ToBeCloseTo(a.sub(0, 0, 0), 3, 3, 3);
        expectVec3ToBeCloseTo(b.sub(2), -4, -2, 0);
    });

    test("scale", () => {
        const a: Vec3 = new Vec3(4, 5, 6);
        const b: Vec3 = new Vec3(1, 2, 3);
        expectVec3ToBeCloseTo(a.scale(b), 4, 10, 18);
        expectVec3ToBeCloseTo(b.scale(3, 2, 1), 3, 4, 3);
        expectVec3ToBeCloseTo(a.scale(2), 8, 20, 36);
    });

    test("divide", () => {
        const a: Vec3 = new Vec3(4, 5, 6);
        const b: Vec3 = new Vec3(1, 2, 3);
        expectVec3ToBeCloseTo(a.divide(b), 4, 2.5, 2);
        expectVec3ToBeCloseTo(b.divide(2, 2, 4), 0.5, 1, 0.75);
        expectVec3ToBeCloseTo(a.divide(2), 2, 1.25, 1);
    });

    test("lengthQuadratic", () => {
        const a: Vec3 = new Vec3(4, 5, 3);
        expect(a.lengthQuadratic()).toBeCloseTo(50);
    });

    test("length", () => {
        const a: Vec3 = new Vec3(4, 5, 3);
        expect(a.length()).toBeCloseTo(7.07);
    });

    test("normalize", () => {
        expectVec3ToBeCloseTo(
            new Vec3(4, 5, 3).normalize(),
            0.565,
            0.707,
            0.424,
        );
    });

    test("dot", () => {
        const a: Vec3 = new Vec3(4, 5, 6);
        const b: Vec3 = new Vec3(1, 2, 3);
        expect(a.dot(b)).toBeCloseTo(32);
        expect(b.dot(2, 2, 4)).toBeCloseTo(18);
        expect(a.dot(2)).toBeCloseTo(30);
    });

    test("cross", () => {
        const a: Vec3 = new Vec3(4, 5, 6);
        const b: Vec3 = new Vec3(1, 2, 3);
        const c: Vec3 = new Vec3(9, 8, 7);
        expectVec3ToBeCloseTo(a.cross(b), 3, -6, 3);
        expectVec3ToBeCloseTo(a.cross(c, b), -10, 20, -10);
    });

    test("applyMat", () => {
        // prettier-ignore
        const a: Mat4 = new Mat4().set(
            3, 4, 2, -6,
            72, 0.5, -3, 62,
            -12, 6, 2, 0.5,
            0.1, -5, 8, 92
        );
        expectVec3ToBeCloseTo(
            new Vec3(4, 5, 6).applyMat(a),
            0.787,
            0.096,
            0.055,
        );
    });

    test("copy", () => {
        const a: Vec3 = new Vec3(4, 5, 6);
        const b: Vec3 = new Vec3(1, 2, 3);
        expect(a.copy(b)).toEqual(b);
    });

    test("store", () => {
        const a: Float32Array = new Float32Array([3, 2, 0, 7, 5]);
        new Vec3(4, 5, 6).store(a, 1);
        expectFloatArrayToBeCloseTo(a, [3, 4, 5, 6, 5]);
    });

    test("clone", () => {
        const a: Vec3 = new Vec3(4, 5, 6);
        expect(a.clone()).toEqual(a);
    });
});
