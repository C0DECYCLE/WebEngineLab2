/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger
 */

import { int, float, Nullable, FloatArray } from "./utils.type.js";
import { Mat2 } from "./Mat2.js";
import { Mat4 } from "./Mat4.js";
import { Vec2 } from "./Vec2.js";
import { normalizeRadian } from "./utils.js";

export class Mat3 {
    public static Cache: Mat3 = new Mat3();

    public readonly values: Float32Array;

    public constructor() {
        this.values = new Float32Array(9);
        this.reset();
    }

    public set(...floats: float[]): Mat3 {
        this.values[0] = floats[0];
        this.values[1] = floats[1];
        this.values[2] = floats[2];
        this.values[3] = floats[3];
        this.values[4] = floats[4];
        this.values[5] = floats[5];
        this.values[6] = floats[6];
        this.values[7] = floats[7];
        this.values[8] = floats[8];
        return this;
    }

    public reset(): Mat3 {
        // prettier-ignore
        return this.set(
            1, 0, 0,
            0, 1, 0,
            0, 0, 1
        );
    }

    public translate(x: Vec2 | float, y: Nullable<float> = null): Mat3 {
        if (x instanceof Vec2) {
            y = x.y;
            x = x.x;
        } else if (y === null) {
            y = x;
        }
        const v: Float32Array = this.values;
        // prettier-ignore
        return this.set(
            v[0], v[1], v[2],
            v[3], v[4], v[5],
            v[0] * x + v[3] * y + v[6], v[1] * x + v[4] * y + v[7], v[2] * x + v[5] * y + v[8]
        );
    }

    public rotate(radian: float): Mat3 {
        radian = normalizeRadian(radian);
        const c: float = Math.cos(radian);
        const s: float = Math.sin(radian);
        const v: Float32Array = this.values;
        // prettier-ignore
        return this.set(
            c * v[0] + s * v[3], c * v[1] + s * v[4], c * v[2] + s * v[5],
            c * v[3] - s * v[0], c * v[4] - s * v[1], c * v[5] - s * v[2],
            v[6], v[7], v[8],
        );
    }

    public scale(x: Vec2 | float, y: Nullable<float> = null): Mat3 {
        if (x instanceof Vec2) {
            y = x.y;
            x = x.x;
        } else if (y === null) {
            y = x;
        }
        const v: Float32Array = this.values;
        // prettier-ignore
        return this.set(
            v[0] * x, v[1] * x, v[2] * x,
            v[3] * y, v[4] * y, v[5] * y,
            v[6], v[7], v[8],
        );
    }

    public transpose(): Mat3 {
        const v: Float32Array = this.values;
        // prettier-ignore
        return this.set(
            v[0],v[3], v[6],
            v[1],v[4], v[7],
            v[2],v[5], v[8],
        );
    }

    public multiply(b: Mat3, a: Mat3 = this): Mat3 {
        const av: Float32Array = a.values;
        const bv: Float32Array = b.values;
        // prettier-ignore
        return this.set(
            av[0] * bv[0] + av[3] * bv[1] + bv[6] * bv[2], av[1] * bv[0] + av[4] * bv[1] + bv[7] * bv[2], av[2] * bv[0] + av[5] * bv[1] + bv[8] * bv[2],
            av[0] * bv[3] + av[3] * bv[4] + bv[6] * bv[5], av[1] * bv[3] + av[4] * bv[4] + bv[7] * bv[5], av[2] * bv[3] + av[5] * bv[4] + bv[8] * bv[5],
            av[0] * bv[6] + av[3] * bv[7] + bv[6] * bv[8], av[1] * bv[6] + av[4] * bv[7] + bv[7] * bv[8], av[2] * bv[6] + av[5] * bv[7] + bv[8] * bv[8],
        );
    }

    public invert(): Mat3 {
        const v: Float32Array = this.values;
        const t01: float = v[8] * v[4] - v[5] * v[7];
        const t11: float = -v[8] * v[3] + v[5] * v[6];
        const t21: float = v[7] * v[3] - v[4] * v[6];
        const det: float = v[0] * t01 + v[1] * t11 + v[2] * t21;
        if (det === 0) {
            // prettier-ignore
            return this.set(
                0, 0, 0,
                0, 0, 0,
                0, 0, 0,
            );
        }
        const detInv: float = 1 / det;
        // prettier-ignore
        return this.set(
            t01 * detInv, (-v[8] * v[1] + v[2] * v[7]) * detInv, (v[5] * v[1] - v[2] * v[4]) * detInv,
            t11 * detInv, (v[8] * v[0] - v[2] * v[6]) * detInv, (-v[5] * v[0] + v[2] * v[3]) * detInv,
            t21 * detInv, (-v[7] * v[0] + v[1] * v[6]) * detInv, (v[4] * v[0] - v[1] * v[3]) * detInv,
        );
    }

    public projection(width: int, height: int): Mat3 {
        // prettier-ignore
        return this.set(
            2 / width, 0, 0,
            0, -2 / height, 0,
            -1, 1, 1,
        );
    }

    public copy(mat: Mat3): Mat3 {
        return this.set(...mat.values);
    }

    public store(
        target: FloatArray,
        offset: int = 0,
        spacing: boolean = false,
    ): Mat3 {
        if (spacing) {
            target[offset] = this.values[0];
            target[offset + 1] = this.values[1];
            target[offset + 2] = this.values[2];
            target[offset + 3] = 0;
            target[offset + 4] = this.values[3];
            target[offset + 5] = this.values[4];
            target[offset + 6] = this.values[5];
            target[offset + 7] = 0;
            target[offset + 8] = this.values[6];
            target[offset + 9] = this.values[7];
            target[offset + 10] = this.values[8];
            target[offset + 11] = 0;
            return this;
        }
        target[offset] = this.values[0];
        target[offset + 1] = this.values[1];
        target[offset + 2] = this.values[2];
        target[offset + 3] = this.values[3];
        target[offset + 4] = this.values[4];
        target[offset + 5] = this.values[5];
        target[offset + 6] = this.values[6];
        target[offset + 7] = this.values[7];
        target[offset + 8] = this.values[8];
        return this;
    }

    public clone(): Mat3 {
        return new Mat3().copy(this);
    }

    public static Projection(width: int, height: int): Mat3 {
        return new Mat3().projection(width, height);
    }

    public static From(mat: Mat2 | Mat3 | Mat4): Mat3 {
        const v: Float32Array = mat.values;
        if (mat instanceof Mat2) {
            // prettier-ignore
            return new Mat3().set(
                v[0], v[1], 0,
                v[2], v[3], 0,
                0, 0, 0,
            );
        }
        if (mat instanceof Mat3) {
            return mat.clone();
        }
        // prettier-ignore
        return new Mat3().set(
            v[0], v[1], v[2],
            v[4], v[5], v[6],
            v[8], v[9], v[10],
        );
    }
}
