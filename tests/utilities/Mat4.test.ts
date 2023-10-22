/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, October 2023
 */

import { float } from "../../types/utilities/utils.type.js";
import { Vec3 } from "../../src/utilities/Vec3.js";
import { Mat4 } from "../../src/utilities/Mat4.js";
import { expectFloatArrayToBeCloseTo } from "../utils.js";

describe("Mat4", () => {
    test("float32", () => {
        const a: Mat4 = new Mat4();
        expect(a.isFloat64).toBe(false);
        expect(a.values).toBeInstanceOf(Float32Array);
        expectFloatArrayToBeCloseTo(
            a.values,
            [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        );
    });

    test("float64", () => {
        const a: Mat4 = new Mat4(true);
        expect(a.isFloat64).toBe(true);
        expect(a.values).toBeInstanceOf(Float64Array);
        expectFloatArrayToBeCloseTo(
            a.values,
            [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        );
    });

    test("set", () => {
        // prettier-ignore
        const a: float[] = [3, 4, 2, -6, 72, 0.5, -3, 62, -12, 6, 2, 0.5, 0.1, -5, 8, 92];
        const b: Mat4 = new Mat4().set(...a);
        expectFloatArrayToBeCloseTo(b.values, a);
    });

    test("reset", () => {
        // prettier-ignore
        const a: float[] = [3, 4, 2, -6, 72, 0.5, -3, 62, -12, 6, 2, 0.5, 0.1, -5, 8, 92];
        const b: Mat4 = new Mat4().set(...a);
        expectFloatArrayToBeCloseTo(
            b.reset().values,
            [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        );
    });

    test("translate", () => {
        // prettier-ignore
        const a: Mat4 = new Mat4().set(3, 4, 2, -6, 72, 5, -3, 62, -12, 6, 2, 5, 1, -5, 8, 92);
        expectFloatArrayToBeCloseTo(
            a.translate(new Vec3(3, 4, 2)).values,
            // prettier-ignore
            [3, 4, 2, -6, 72, 5, -3, 62, -12, 6, 2, 5, 274, 39, 6, 332],
        );
        expectFloatArrayToBeCloseTo(
            a.translate(-2).values,
            // prettier-ignore
            [3, 4, 2, -6, 72, 5, -3, 62, -12, 6, 2, 5, 148, 9, 4, 210],
        );
        expectFloatArrayToBeCloseTo(
            a.translate(1, 2, 3).values,
            // prettier-ignore
            [3, 4, 2, -6, 72, 5, -3, 62, -12, 6, 2, 5, 259, 41, 6, 343],
        );
        expectFloatArrayToBeCloseTo(
            a.translate(0).values,
            // prettier-ignore
            [3, 4, 2, -6, 72, 5, -3, 62, -12, 6, 2, 5, 259, 41, 6, 343],
        );
    });

    test("rotateX", () => {
        // prettier-ignore
        const a: Mat4 = new Mat4().set(3, 4, 2, -6, 72, 5, -3, 62, -12, 6, 2, 5, 1, -5, 8, 92);
        expectFloatArrayToBeCloseTo(
            a.rotateX(2.356).values,
            // prettier-ignore
            [3, 4, 2, -6, -59.388, 0.708, 3.535, -40.295, -42.437, -7.778, 0.707, -47.383, 1, -5, 8, 92],
        );
        expectFloatArrayToBeCloseTo(
            a.rotateX(0).values,
            // prettier-ignore
            [3, 4, 2, -6, -59.388, 0.708, 3.535, -40.295, -42.437, -7.778, 0.707, -47.383, 1, -5, 8, 92],
        );
    });

    test("rotateY", () => {
        // prettier-ignore
        const a: Mat4 = new Mat4().set(3, 4, 2, -6, 72, 5, -3, 62, -12, 6, 2, 5, 1, -5, 8, 92);
        expectFloatArrayToBeCloseTo(
            a.rotateY(2.356).values,
            // prettier-ignore
            [6.366, -7.071, -2.828, 0.705, 72, 5, -3, 62, 10.605, -1.412, 0.000, -7.778, 1, -5, 8, 92],
        );
        expectFloatArrayToBeCloseTo(
            a.rotateY(0).values,
            // prettier-ignore
            [6.366, -7.071, -2.828, 0.705, 72, 5, -3, 62, 10.605, -1.412, 0.000, -7.778, 1, -5, 8, 92],
        );
    });

    test("rotateZ", () => {
        // prettier-ignore
        const a: Mat4 = new Mat4().set(3, 4, 2, -6, 72, 5, -3, 62, -12, 6, 2, 5, 1, -5, 8, 92);
        expectFloatArrayToBeCloseTo(
            a.rotateZ(2.356).values,
            // prettier-ignore
            [48.800, 0.708, -3.535, 48.090, -53.023, -6.363, 0.706, -39.588, -12, 6, 2, 5, 1, -5, 8, 92],
        );
        expectFloatArrayToBeCloseTo(
            a.rotateZ(0).values,
            // prettier-ignore
            [48.800, 0.708, -3.535, 48.090, -53.023, -6.363, 0.706, -39.588, -12, 6, 2, 5, 1, -5, 8, 92],
        );
    });

    test("rotateAxis", () => {
        // prettier-ignore
        const a: Mat4 = new Mat4().set(3, 4, 2, -6, 72, 5, -3, 62, -12, 6, 2, 5, 1, -5, 8, 92);
        const b: Vec3 = new Vec3(0.565, -0.707, 0.424);
        expectFloatArrayToBeCloseTo(
            a.rotateAxis(b, 2.356).values,
            // prettier-ignore
            [-38.896, 2.895, 2.639, -18.166, 8.930, -3.868, -2.626, 14.397, -61.084, -7.321, 1.751, -57.953, 1, -5, 8, 92],
        );
        expectFloatArrayToBeCloseTo(
            a.rotateAxis(b, 0).values,
            // prettier-ignore
            [-38.896, 2.895, 2.639, -18.166, 8.930, -3.868, -2.626, 14.397, -61.084, -7.321, 1.751, -57.953, 1, -5, 8, 92],
        );
    });

    test("scale", () => {
        // prettier-ignore
        const a: Mat4 = new Mat4().set(3, 4, 2, -6, 72, 5, -3, 62, -12, 6, 2, 5, 1, -5, 8, 92);
        expectFloatArrayToBeCloseTo(
            a.scale(new Vec3(3, 4, 2)).values,
            // prettier-ignore
            [9, 12, 6, -18, 288, 20, -12, 248, -24, 12, 4, 10, 1, -5, 8, 92],
        );
        expectFloatArrayToBeCloseTo(
            a.scale(-2).values,
            // prettier-ignore
            [-18, -24, -12, 36, -576, -40, 24, -496, 48, -24, -8, -20, 1, -5, 8, 92],
        );
        expectFloatArrayToBeCloseTo(
            a.scale(1, 2, 3).values,
            // prettier-ignore
            [-18, -24, -12, 36, -1152, -80, 48, -992, 144, -72, -24, -60, 1, -5, 8, 92],
        );
        expectFloatArrayToBeCloseTo(
            a.scale(0).values,
            // prettier-ignore
            [-18, -24, -12, 36, -1152, -80, 48, -992, 144, -72, -24, -60, 1, -5, 8, 92],
        );
    });

    test("multiply", () => {
        // prettier-ignore
        const a: Mat4 = new Mat4().set(3, 4, 2, -6, 72, 5, -3, 62, -12, 6, 2, 5, 1, -5, 8, 92);
        // prettier-ignore
        const b: Mat4 = new Mat4().set(73, 0, 3, -6, 72, 5, -1, 2, -5, 9, 2, 5, 1, 4, 8, 6);
        // prettier-ignore
        const c: Mat4 = new Mat4().set(9, 12, 6, -18, 288, 20, -12, 248, -24, 12, 4, 10, 1, -5, 8, 92);
        expectFloatArrayToBeCloseTo(
            a.multiply(b).values,
            // prettier-ignore
            [177, 340, 104, -975, 590, 297, 143, 57, 614, 12, 7, 1058, 201, 42, 54, 834],
        );
        expectFloatArrayToBeCloseTo(
            a.multiply(c, b).values,
            // prettier-ignore
            [1473, 42, -117, -108, 22772, 984, 2804, -260, -898, 136, 4, 248, -235, 415, 760, 576],
        );
    });

    test("invert", () => {
        // prettier-ignore
        const a: Mat4 = new Mat4().set(3, 4, 2, -6, 72, 5, -3, 62, -12, 6, 2, 5, 1, -5, 8, 92);
        expectFloatArrayToBeCloseTo(
            a.invert().values,
            // prettier-ignore
            [0.042, 0.006, -0.033, 0.000, -0.006, 0.022, 0.133, -0.023, 0.354, -0.041, -0.152, 0.059, -0.031, 0.004, 0.020, 0.004],
        );
    });

    test("aim", () => {
        expectFloatArrayToBeCloseTo(
            new Mat4().aim(
                new Vec3(52, 71, 23),
                new Vec3(-5, 2, 1).normalize(),
                new Vec3(1, 6, -4).normalize(),
            ).values,
            // prettier-ignore
            [0.352, 0.477, 0.804, 0, 0.206, 0.798, -0.564, 0, -0.912, 0.365, 0.182, 0, 52, 71, 23, 1],
        );
    });

    test("view", () => {
        expectFloatArrayToBeCloseTo(
            new Mat4().view(
                new Vec3(52, 71, 23),
                new Vec3(-5, 2, 1).normalize(),
                new Vec3(1, 6, -4).normalize(),
            ).values,
            // prettier-ignore
            [0.352, 0.206, -0.912, 0, 0.477, 0.798, 0.365, 0, 0.804, -0.564, 0.182, 0, -70.746, -54.480, 17.344, 1],
        );
    });

    test("perspective", () => {
        expectFloatArrayToBeCloseTo(
            new Mat4().perspective(
                60 * (Math.PI / 180),
                1920 / 1080,
                0.01,
                1000.0,
            ).values,
            // prettier-ignore
            [0.974, 0, 0, 0, 0, 1.732, 0, 0, 0, 0, -1.000, -1, 0, 0, -0.010, 0],
        );
    });

    test("copy", () => {
        // prettier-ignore
        const a: Mat4 = new Mat4().set(3, 4, 2, -6, 72, 5, -3, 62, -12, 6, 2, 5, 1, -5, 8, 92);
        // prettier-ignore
        const b: Mat4 = new Mat4().set(73, 0, 3, -6, 72, 5, -1, 2, -5, 9, 2, 5, 1, 4, 8, 6);
        expect(a.copy(b)).toEqual(b);
    });

    test("store", () => {
        const a: Float32Array = new Float32Array([
            3, 4, 2, -6, 72, 5, -3, 62, -12, 6, 2, 5, 1, -5, 8, 92, 3, 4, 2, -6,
            72, 5, -3, 62, -12, 6, 2, 5, 1, -5, 8, 92, 5, -3, 62, -12, 6, 10, 0,
        ]);
        new Mat4()
            .set(3, 4, 2, -6, 72, 5, -3, 62, -12, 6, 2, 5, 1, -5, 8, 92)
            .store(a, 7);
        expectFloatArrayToBeCloseTo(
            a,
            [
                3, 4, 2, -6, 72, 5, -3, 3, 4, 2, -6, 72, 5, -3, 62, -12, 6, 2,
                5, 1, -5, 8, 92, 62, -12, 6, 2, 5, 1, -5, 8, 92, 5, -3, 62, -12,
                6, 10, 0,
            ],
        );
    });

    test("clone", () => {
        // prettier-ignore
        const a: Mat4 = new Mat4().set(3, 4, 2, -6, 72, 5, -3, 62, -12, 6, 2, 5, 1, -5, 8, 92);
        expect(a.clone()).toEqual(a);
    });
});
