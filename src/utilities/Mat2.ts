/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger
 */

import { int, float, Nullable, FloatArray } from "./utils.type.js";
import { Mat3 } from "./Mat3.js";
import { Mat4 } from "./Mat4.js";
import { Vec2 } from "./Vec2.js";
import { normalizeRadian } from "./utils.js";

export class Mat2 {
    public static Cache: Mat2 = new Mat2();

    public readonly values: Float32Array;

    public constructor() {
        this.values = new Float32Array(4);
        this.reset();
    }

    public set(...floats: float[]): Mat2 {
        this.values[0] = floats[0];
        this.values[1] = floats[1];
        this.values[2] = floats[2];
        this.values[3] = floats[3];
        return this;
    }

    public reset(): Mat2 {
        // prettier-ignore
        return this.set(
            1, 0,
            0, 1,
        );
    }

    public rotate(radian: float): Mat2 {
        radian = normalizeRadian(radian);
        const c: float = Math.cos(radian);
        const s: float = Math.sin(radian);
        const v: Float32Array = this.values;
        // prettier-ignore
        return this.set(
            c * v[0] + s * v[2], c * v[1] + s * v[3],
            c * v[2] - s * v[0], c * v[3] - s * v[1],
        );
    }

    public scale(x: Vec2 | float, y: Nullable<float> = null): Mat2 {
        if (x instanceof Vec2) {
            y = x.y;
            x = x.x;
        } else if (y === null) {
            y = x;
        }
        const v: Float32Array = this.values;
        // prettier-ignore
        return this.set(
            v[0] * x, v[1] * x,
            v[2] * y, v[3] * y,
        );
    }

    public transpose(): Mat2 {
        const v: Float32Array = this.values;
        // prettier-ignore
        return this.set(
            v[0],v[2],
            v[1],v[3],
        );
    }

    public multiply(b: Mat2, a: Mat2 = this): Mat2 {
        const av: Float32Array = a.values;
        const bv: Float32Array = b.values;
        // prettier-ignore
        return this.set(
            av[0] * bv[0] + av[2] * bv[1], av[1] * bv[0] + av[3] * bv[1],
            av[0] * bv[2] + av[2] * bv[3], av[1] * bv[2] + av[3] * bv[3],
        );
    }

    public invert(): Mat2 {
        const v: Float32Array = this.values;
        const det: float = v[0] * v[3] - v[2] * v[1];
        if (det === 0) {
            // prettier-ignore
            return this.set(
                0, 0,
                0, 0,
            );
        }
        const detInv: float = 1 / det;
        // prettier-ignore
        return this.set(
            v[3] * detInv, -v[1] * detInv,
            -v[2] * detInv, v[0] * detInv,
        );
    }

    public copy(mat: Mat2): Mat2 {
        return this.set(...mat.values);
    }

    public store(target: FloatArray, offset: int = 0): Mat2 {
        target[offset] = this.values[0];
        target[offset + 1] = this.values[1];
        target[offset + 2] = this.values[2];
        target[offset + 3] = this.values[3];
        return this;
    }

    public clone(): Mat2 {
        return new Mat2().copy(this);
    }

    public static From(mat: Mat2 | Mat3 | Mat4): Mat2 {
        const v: Float32Array = mat.values;
        if (mat instanceof Mat2) {
            return mat.clone();
        }
        if (mat instanceof Mat3) {
            // prettier-ignore
            return new Mat2().set(
                v[0], v[1],
                v[3], v[4],
            );
        }
        // prettier-ignore
        return new Mat2().set(
            v[0], v[1],
            v[4], v[5],
        );
    }
}
