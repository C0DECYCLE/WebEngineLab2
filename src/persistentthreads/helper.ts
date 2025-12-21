/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger
 */

import { log } from "../utilities/logger.js";
import { assert } from "../utilities/utils.js";
import { int } from "../utilities/utils.type.js";
import { InputLayerCount, InputMax, InputSize } from "./index.js";

export function populate(empty: Uint32Array): void {
    for (let i: int = 0; i < InputSize; i++) {
        //don't allow zero to avoid workaround in persistent thread
        empty[i] = Math.floor(Math.random() * (InputMax - 1)) + 1;
    }
}

export function simulate(raw: Uint32Array): Uint32Array {
    const out: Uint32Array = new Uint32Array(InputSize * 2);
    for (let i: int = 0; i < InputSize; i++) {
        out[i] = raw[i];
    }
    const pre: int = performance.now();
    for (let i: int = 0; i < InputSize - 1; i++) {
        const a: int = out[i * 2 + 0];
        const b: int = out[i * 2 + 1];
        out[InputSize + i] = a + b;
    }
    const post: int = performance.now();
    log("Simulation CPU time (ms)", post - pre);
    return out;
}

export function ensure(a: Uint32Array, b: Uint32Array, length: int): void {
    for (let i: int = 0; i < length; i++) {
        assert(a[i] === b[i]);
    }
}

export function output(data: Uint32Array, resultOnly: boolean): void {
    if (resultOnly) {
        log("Result", data[InputSize * 2 - 2]);
        return;
    }
    for (let i: int = 0; i < InputLayerCount + 1; i++) {
        let offset: int = 0;
        for (let o: int = 0; o < i; o++) {
            offset += Math.pow(2, InputLayerCount - o);
        }
        const length: int = Math.pow(2, InputLayerCount - i);
        log(data.slice(offset, offset + length).join(" "));
    }
}

export function verify(a: Uint32Array, b: Uint32Array): void {
    assert(a.length === b.length);
    const length: int = a.length;
    ensure(a, b, length);
    assert(a[length - 2] === b[length - 2]);
}
