/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, October 2023
 */

import { float, Nullable } from "../types/utilities/utils.type.js";
import { Stats } from "../src/Stats.js";

describe("Stats", () => {
    let stats: Nullable<Stats> = null;

    beforeEach(() => {
        stats = new Stats();
    });

    afterEach(() => {
        stats?.destroy();
        stats = null;
    });

    test("get", () => {
        const key: string = "abc";
        expect(stats).not.toBeNull();
        expect(stats?.get(key)).toBeUndefined();
    });

    test("set", () => {
        const key: string = "abc";
        const value: float = 1234.5678;
        expect(stats).not.toBeNull();
        expect(stats?.set(key, value)).toBeUndefined();
        expect(stats?.get(key)).toBeCloseTo(value);
    });

    test("add", () => {
        const key: string = "abc";
        const value: float = 1234.5678;
        expect(stats).not.toBeNull();
        expect(stats?.set(key, value)).toBeUndefined();
        expect(stats?.add(key, value)).toBeUndefined();
        expect(stats?.get(key)).toBeCloseTo(2 * value);
    });

    test("time", () => {
        const key: string = "abc";
        expect(stats).not.toBeNull();
        expect(stats?.get(key)).toBeUndefined();
        expect(stats?.time(key)).toBeUndefined();
        expect(stats?.get(key)).not.toBeUndefined();
        expect(stats?.time(key, key)).toBeUndefined();
        expect(stats?.get(key)).not.toBeUndefined();
    });
});
