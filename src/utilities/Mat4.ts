/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger
 */

import { int, float, Nullable, FloatArray } from "./utils.type.js";
import { Mat2 } from "./Mat2.js";
import { Mat3 } from "./Mat3.js";
import { Vec3 } from "./Vec3.js";
import { normalizeRadian } from "./utils.js";

export class Mat4 {
    public static Cache: Mat4 = new Mat4();

    public readonly values: Float32Array;

    public constructor() {
        this.values = new Float32Array(16);
        this.reset();
    }

    public set(...floats: float[]): Mat4 {
        this.values[0] = floats[0];
        this.values[1] = floats[1];
        this.values[2] = floats[2];
        this.values[3] = floats[3];
        this.values[4] = floats[4];
        this.values[5] = floats[5];
        this.values[6] = floats[6];
        this.values[7] = floats[7];
        this.values[8] = floats[8];
        this.values[9] = floats[9];
        this.values[10] = floats[10];
        this.values[11] = floats[11];
        this.values[12] = floats[12];
        this.values[13] = floats[13];
        this.values[14] = floats[14];
        this.values[15] = floats[15];
        return this;
    }

    public reset(): Mat4 {
        // prettier-ignore
        return this.set(
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1
        );
    }

    public translate(
        x: Vec3 | float,
        y: Nullable<float> = null,
        z: Nullable<float> = null,
    ): Mat4 {
        if (x instanceof Vec3) {
            z = x.z;
            y = x.y;
            x = x.x;
        } else if (y === null || z === null) {
            z = x;
            y = x;
        }
        const v: Float32Array = this.values;
        // prettier-ignore
        return this.set(
            v[0], v[1], v[2], v[3],
            v[4], v[5], v[6], v[7],
            v[8], v[9], v[10], v[11],
            v[0] * x + v[4] * y + v[8] * z + v[12], v[1] * x + v[5] * y + v[9] * z + v[13], v[2] * x + v[6] * y + v[10] * z + v[14], v[3] * x + v[7] * y + v[11] * z + v[15],
        );
    }

    public rotateX(radian: float): Mat4 {
        radian = normalizeRadian(radian);
        const c: float = Math.cos(radian);
        const s: float = Math.sin(radian);
        const v: Float32Array = this.values;
        // prettier-ignore
        return this.set(
            v[0], v[1], v[2], v[3],
            c * v[4] + s * v[8], c * v[5] + s * v[9], c * v[6] + s * v[10], c * v[7] + s * v[11],
            c * v[8] - s * v[4], c * v[9] - s * v[5], c * v[10] - s * v[6], c * v[11] - s * v[7],
            v[12], v[13], v[14], v[15],
        );
    }

    public rotateY(radian: float): Mat4 {
        radian = normalizeRadian(radian);
        const c: float = Math.cos(radian);
        const s: float = Math.sin(radian);
        const v: Float32Array = this.values;
        // prettier-ignore
        return this.set(
            c * v[0] - s * v[8], c * v[1] - s * v[9], c * v[2] - s * v[10], c * v[3] - s * v[11],
            v[4], v[5], v[6], v[7],
            c * v[8] + s * v[0], c * v[9] + s * v[1], c * v[10] + s * v[2], c * v[11] + s * v[3],
            v[12], v[13], v[14], v[15],
        );
    }

    public rotateZ(radian: float): Mat4 {
        radian = normalizeRadian(radian);
        const c: float = Math.cos(radian);
        const s: float = Math.sin(radian);
        const v: Float32Array = this.values;
        // prettier-ignore
        return this.set(
            c * v[0] + s * v[4], c * v[1] + s * v[5], c * v[2] + s * v[6], c * v[3] + s * v[7],
            c * v[4] - s * v[0], c * v[5] - s * v[1], c * v[6] - s * v[2], c * v[7] - s * v[3],
            v[8], v[9], v[10], v[11],
            v[12], v[13], v[14], v[15],
        );
    }

    public rotateAxis(normal: Vec3, radian: float): Mat4 {
        radian = normalizeRadian(radian);
        const xx: float = normal.x * normal.x;
        const yy: float = normal.y * normal.y;
        const zz: float = normal.z * normal.z;
        const c: float = Math.cos(radian);
        const cI: float = 1.0 - c;
        const s: float = Math.sin(radian);
        // prettier-ignore
        const rot: Mat3 = new Mat3().set(
            xx + (1 - xx) * c, normal.x * normal.y * cI + normal.z * s, normal.x * normal.z * cI - normal.y * s,
            normal.x * normal.y * cI - normal.z * s, yy + (1 - yy) * c, normal.y * normal.z * cI + normal.x * s,
            normal.x * normal.z * cI + normal.y * s, normal.y * normal.z * cI - normal.x * s, zz + (1 - zz) * c,
        );
        const r: Float32Array = rot.values;
        const v: Float32Array = this.values;
        // prettier-ignore
        return this.set(
            r[0] * v[0] + r[1] * v[4] + r[2] * v[8], r[0] * v[1] + r[1] * v[5] + r[2] * v[9], r[0] * v[2] + r[1] * v[6] + r[2] * v[10], r[0] * v[3] + r[1] * v[7] + r[2] * v[11],
            r[3] * v[0] + r[4] * v[4] + r[5] * v[8], r[3] * v[1] + r[4] * v[5] + r[5] * v[9], r[3] * v[2] + r[4] * v[6] + r[5] * v[10], r[3] * v[3] + r[4] * v[7] + r[5] * v[11],
            r[6] * v[0] + r[7] * v[4] + r[8] * v[8], r[6] * v[1] + r[7] * v[5] + r[8] * v[9], r[6] * v[2] + r[7] * v[6] + r[8] * v[10], r[6] * v[3] + r[7] * v[7] + r[8] * v[11],
            v[12], v[13], v[14], v[15],
        );
    }

    public scale(
        x: Vec3 | float,
        y: Nullable<float> = null,
        z: Nullable<float> = null,
    ): Mat4 {
        if (x instanceof Vec3) {
            z = x.z;
            y = x.y;
            x = x.x;
        } else if (y === null || z === null) {
            y = x;
            z = x;
        }
        const v: Float32Array = this.values;
        // prettier-ignore
        return this.set(
            v[0] * x, v[1] * x, v[2] * x, v[3] * x,
            v[4] * y, v[5] * y, v[6] * y, v[7] * y,
            v[8] * z, v[9] * z, v[10] * z, v[11] * z,
            v[12], v[13], v[14], v[15],
        );
    }

    public transpose(): Mat4 {
        const v: Float32Array = this.values;
        // prettier-ignore
        return this.set(
            v[0],v[4], v[8], v[12],
            v[1],v[5], v[9], v[13],
            v[2],v[6], v[10], v[14],
            v[3],v[7], v[11], v[15],
        );
    }

    public multiply(b: Mat4, a: Mat4 = this): Mat4 {
        const av: Float32Array = a.values;
        const bv: Float32Array = b.values;
        // prettier-ignore
        return this.set(
            bv[0] * av[0] + bv[1] * av[4] + bv[2] * av[8] + bv[3] * av[12], bv[0] * av[1] + bv[1] * av[5] + bv[2] * av[9] + bv[3] * av[13], bv[0] * av[2] + bv[1] * av[6] + bv[2] * av[10] + bv[3] * av[14], bv[0] * av[3] + bv[1] * av[7] + bv[2] * av[11] + bv[3] * av[15],
            bv[4] * av[0] + bv[5] * av[4] + bv[6] * av[8] + bv[7] * av[12], bv[4] * av[1] + bv[5] * av[5] + bv[6] * av[9] + bv[7] * av[13], bv[4] * av[2] + bv[5] * av[6] + bv[6] * av[10] + bv[7] * av[14], bv[4] * av[3] + bv[5] * av[7] + bv[6] * av[11] + bv[7] * av[15],
            bv[8] * av[0] + bv[9] * av[4] + bv[10] * av[8] + bv[11] * av[12], bv[8] * av[1] + bv[9] * av[5] + bv[10] * av[9] + bv[11] * av[13], bv[8] * av[2] + bv[9] * av[6] + bv[10] * av[10] + bv[11] * av[14], bv[8] * av[3] + bv[9] * av[7] + bv[10] * av[11] + bv[11] * av[15],
            bv[12] * av[0] + bv[13] * av[4] + bv[14] * av[8] + bv[15] * av[12], bv[12] * av[1] + bv[13] * av[5] + bv[14] * av[9] + bv[15] * av[13], bv[12] * av[2] + bv[13] * av[6] + bv[14] * av[10] + bv[15] * av[14], bv[12] * av[3] + bv[13] * av[7] + bv[14] * av[11] + bv[15] * av[15],
        );
    }

    public invert(): Mat4 {
        const v: Float32Array = this.values;
        // prettier-ignore
        const t01: float = v[9] * v[14] * v[7] - v[13] * v[10] * v[7] + v[13] * v[6] * v[11] - v[5] * v[14] * v[11] - v[9] * v[6] * v[15] + v[5] * v[10] * v[15];
        // prettier-ignore
        const t11: float = v[12] * v[10] * v[7] - v[8] * v[14] * v[7] - v[12] * v[6] * v[11] + v[4] * v[14] * v[11] + v[8] * v[6] * v[15] - v[4] * v[10] * v[15];
        // prettier-ignore
        const t21: float = v[8] * v[13] * v[7] - v[12] * v[9] * v[7] + v[12] * v[5] * v[11] - v[4] * v[13] * v[11] - v[8] * v[5] * v[15] + v[4] * v[9] * v[15];
        // prettier-ignore
        const t31: float = v[12] * v[9] * v[6] - v[8] * v[13] * v[6] - v[12] * v[5] * v[10] + v[4] * v[13] * v[10] + v[8] * v[5] * v[14] - v[4] * v[9] * v[14];
        const det: float = v[0] * t01 + v[1] * t11 + v[2] * t21 + v[3] * t31;
        if (det === 0) {
            // prettier-ignore
            return this.set(
                0, 0, 0, 0,
                0, 0, 0, 0,
                0, 0, 0, 0,
                0, 0, 0, 0,
            );
        }
        const detInv: float = 1 / det;
        // prettier-ignore
        return this.set(
            t01 * detInv, (v[13] * v[10] * v[3] - v[9] * v[14] * v[3] - v[13] * v[2] * v[11] + v[1] * v[14] * v[11] + v[9] * v[2] * v[15] - v[1] * v[10] * v[15]) * detInv, (v[5] * v[14] * v[3] - v[13] * v[6] * v[3] + v[13] * v[2] * v[7] - v[1] * v[14] * v[7] - v[5] * v[2] * v[15] + v[1] * v[6] * v[15]) * detInv, (v[9] * v[6] * v[3] - v[5] * v[10] * v[3] - v[9] * v[2] * v[7] + v[1] * v[10] * v[7] + v[5] * v[2] * v[11] - v[1] * v[6] * v[11]) * detInv,
            t11 * detInv, (v[8] * v[14] * v[3] - v[12] * v[10] * v[3] + v[12] * v[2] * v[11] - v[0] * v[14] * v[11] - v[8] * v[2] * v[15] + v[0] * v[10] * v[15]) * detInv, (v[12] * v[6] * v[3] - v[4] * v[14] * v[3] - v[12] * v[2] * v[7] + v[0] * v[14] * v[7] + v[4] * v[2] * v[15] - v[0] * v[6] * v[15]) * detInv, (v[4] * v[10] * v[3] - v[8] * v[6] * v[3] + v[8] * v[2] * v[7] - v[0] * v[10] * v[7] - v[4] * v[2] * v[11] + v[0] * v[6] * v[11]) * detInv, 
            t21 * detInv, (v[12] * v[9] * v[3] - v[8] * v[13] * v[3] - v[12] * v[1] * v[11] + v[0] * v[13] * v[11] + v[8] * v[1] * v[15] - v[0] * v[9] * v[15]) * detInv, (v[4] * v[13] * v[3] - v[12] * v[5] * v[3] + v[12] * v[1] * v[7] - v[0] * v[13] * v[7] - v[4] * v[1] * v[15] + v[0] * v[5] * v[15]) * detInv, (v[8] * v[5] * v[3] - v[4] * v[9] * v[3] - v[8] * v[1] * v[7] + v[0] * v[9] * v[7] + v[4] * v[1] * v[11] - v[0] * v[5] * v[11]) * detInv, 
            t31 * detInv, (v[8] * v[13] * v[2] - v[12] * v[9] * v[2] + v[12] * v[1] * v[10] - v[0] * v[13] * v[10] - v[8] * v[1] * v[14] + v[0] * v[9] * v[14]) * detInv, (v[12] * v[5] * v[2] - v[4] * v[13] * v[2] - v[12] * v[1] * v[6] + v[0] * v[13] * v[6] + v[4] * v[1] * v[14] - v[0] * v[5] * v[14]) * detInv, (v[4] * v[9] * v[2] - v[8] * v[5] * v[2] + v[8] * v[1] * v[6] - v[0] * v[9] * v[6] - v[4] * v[1] * v[10] + v[0] * v[5] * v[10]) * detInv, 
        );
    }

    public aim(pos: Vec3, dir: Vec3, up: Vec3): Mat4 {
        const xAxis: Vec3 = Vec3.Cross(up, dir).normalize();
        const yAxis: Vec3 = Vec3.Cross(dir, xAxis);
        // prettier-ignore
        return this.set(
            xAxis.x, xAxis.y, xAxis.z, 0,
            yAxis.x, yAxis.y, yAxis.z, 0,
            dir.x, dir.y, dir.z, 0,
            pos.x, pos.y, pos.z, 1
        );
    }

    public view(pos: Vec3, dir: Vec3, up: Vec3): Mat4 {
        const xAxis: Vec3 = Vec3.Cross(up, dir).normalize();
        const yAxis: Vec3 = Vec3.Cross(dir, xAxis);
        const posI: Vec3 = new Vec3(
            -(xAxis.x * pos.x + xAxis.y * pos.y + xAxis.z * pos.z),
            -(yAxis.x * pos.x + yAxis.y * pos.y + yAxis.z * pos.z),
            -(dir.x * pos.x + dir.y * pos.y + dir.z * pos.z),
        );
        // prettier-ignore
        return this.set(
            xAxis.x, yAxis.x, dir.x, 0,
            xAxis.y, yAxis.y, dir.y, 0,
            xAxis.z, yAxis.z, dir.z, 0,
            posI.x, posI.y, posI.z, 1
        );
    }

    public perspective(
        fov: float,
        aspect: float,
        near: float,
        far: float,
    ): Mat4 {
        const f: float = Math.tan(Math.PI * 0.5 - 0.5 * fov);
        const rangeInv: float = 1.0 / (near - far);
        // prettier-ignore
        return this.set(
            f / aspect, 0, 0, 0,
            0, f, 0, 0,
            0, 0, far * rangeInv, -1,
            0, 0, near * far * rangeInv, 0,
        );
    }

    public orthogonal(
        left: float,
        right: float,
        top: float,
        bottom: float,
        near: float,
        far: float,
    ): Mat4 {
        // prettier-ignore
        return this.set(
            2 / (right - left), 0, 0, 0,
            0, 2 / (top - bottom), 0, 0,
            0, 0, 1 / (near - far), 0,
            (right + left) / (left - right), (top + bottom) / (bottom - top), near / (near - far), 1,
        );
    }

    public copy(mat: Mat4): Mat4 {
        return this.set(...mat.values);
    }

    public store(target: FloatArray, offset: int = 0): Mat4 {
        target[offset] = this.values[0];
        target[offset + 1] = this.values[1];
        target[offset + 2] = this.values[2];
        target[offset + 3] = this.values[3];
        target[offset + 4] = this.values[4];
        target[offset + 5] = this.values[5];
        target[offset + 6] = this.values[6];
        target[offset + 7] = this.values[7];
        target[offset + 8] = this.values[8];
        target[offset + 9] = this.values[9];
        target[offset + 10] = this.values[10];
        target[offset + 11] = this.values[11];
        target[offset + 12] = this.values[12];
        target[offset + 13] = this.values[13];
        target[offset + 14] = this.values[14];
        target[offset + 15] = this.values[15];
        return this;
    }

    public clone(): Mat4 {
        return new Mat4().copy(this);
    }

    public static Aim(pos: Vec3, dir: Vec3, up: Vec3): Mat4 {
        return new Mat4().aim(pos, dir, up);
    }

    public static View(pos: Vec3, dir: Vec3, up: Vec3): Mat4 {
        return new Mat4().view(pos, dir, up);
    }

    public static Perspective(
        fov: float,
        aspect: float,
        near: float,
        far: float,
    ): Mat4 {
        return new Mat4().perspective(fov, aspect, near, far);
    }

    public static Orthogonal(
        left: float,
        right: float,
        top: float,
        bottom: float,
        near: float,
        far: float,
    ): Mat4 {
        return new Mat4().orthogonal(left, right, top, bottom, near, far);
    }

    public static From(mat: Mat2 | Mat3 | Mat4): Mat4 {
        const v: Float32Array = mat.values;
        if (mat instanceof Mat2) {
            // prettier-ignore
            return new Mat4().set(
                v[0], v[1], 0, 0,
                v[2], v[3], 0, 0,
                0, 0, 0, 0,
            );
        }
        if (mat instanceof Mat3) {
            // prettier-ignore
            return new Mat4().set(
                v[0], v[1], v[2], 0,
                v[3], v[4], v[5], 0,
                v[6], v[7], v[8], 0,
                0, 0, 0, 0,
            );
        }
        return mat.clone();
    }
}
