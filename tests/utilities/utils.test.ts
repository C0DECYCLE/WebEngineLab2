/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, October 2023
 */

import { int } from "../../types/utilities/utils.type.js";
import {
    PHI,
    UUIDv4,
    between,
    clamp,
    clear,
    count,
    dotit,
    firstLetterUppercase,
    normalizeRadian,
    replaceAt,
    toAngle,
    toHexadecimal,
    toRadian,
} from "../../src/utilities/utils.js";

describe("utils", () => {
    test("PHI", () => {
        expect(PHI).toBeCloseTo(1.618);
    });

    test("toAngle", () => {
        expect(toAngle).toBeCloseTo(57.295);
    });

    test("toRadian", () => {
        expect(toRadian).toBeCloseTo(0.017);
    });

    test("normalizeRadian", () => {
        expect(normalizeRadian(0)).toBeCloseTo(0);
        expect(normalizeRadian(118 * toRadian)).toBeCloseTo(118 * toRadian);
        expect(normalizeRadian(538 * toRadian)).toBeCloseTo(178 * toRadian);
        expect(normalizeRadian(-538 * toRadian)).toBeCloseTo(182 * toRadian);
    });

    test("UUIDv4", () => {
        expect(UUIDv4().length).toBe(36);
        expect(UUIDv4()[8]).toBe("-");
        expect(UUIDv4()[13]).toBe("-");
        expect(UUIDv4()[18]).toBe("-");
    });

    test("between", () => {
        expect(between(8, 7, 10)).toBe(true);
        expect(between(2, 7, 10)).toBe(false);
        expect(between(5.02, 5.015, 5.0201)).toBe(true);
    });

    test("dotit", () => {
        expect(dotit(1239022)).toBe("1'239'022");
        expect(dotit(-1239022.782)).toBe("-1'239'023");
    });

    test("clamp", () => {
        expect(clamp(8, 7, 10)).toBeCloseTo(8);
        expect(clamp(2, 7, 10)).toBeCloseTo(7);
        expect(clamp(12, 7, 10)).toBeCloseTo(10);
    });

    test("firstLetterUppercase", () => {
        expect(firstLetterUppercase("abc de")).toBe("Abc de");
    });

    test("replaceAt", () => {
        expect(replaceAt("abc de", 1, "f")).toBe("afc de");
    });

    test("toHexadecimal", () => {
        expect(toHexadecimal("#ABABAB")).toBe(11250603);
    });

    test("count", () => {
        expect(count<int>([8, 82, 0, 8, 3], 8)).toBeCloseTo(2);
    });

    test("clear", () => {
        expect(clear<int>([8, 82, 0, 8, 3])).toEqual([]);
    });
});
