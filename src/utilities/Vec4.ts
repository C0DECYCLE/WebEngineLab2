/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger
 */

import { int, float, Nullable, FloatArray } from "./utils.type.js";
import { Mat4 } from "./Mat4.js";
import { Vec2Like } from "./Vec2.js";
import { Vec3Like } from "./Vec3.js";

export type Vec4Like = {
    x: float;
    y: float;
    z: float;
    w: float;
};

export class Vec4 {
    public static Cache: Vec4 = new Vec4();

    protected _x!: float;
    protected _y!: float;
    protected _z!: float;
    protected _w!: float;

    public get x(): float {
        return this._x;
    }

    public get y(): float {
        return this._y;
    }

    public get z(): float {
        return this._z;
    }

    public get w(): float {
        return this._w;
    }

    public set x(value: float) {
        this._x = value;
    }

    public set y(value: float) {
        this._y = value;
    }

    public set z(value: float) {
        this._z = value;
    }

    public set w(value: float) {
        this._w = value;
    }

    public constructor(
        x: Vec4Like | float = 0,
        y: float = 0,
        z: float = 0,
        w: float = 0,
    ) {
        this.set(x, y, z, w);
    }

    public set(
        x: Vec4Like | float,
        y: Nullable<float> = null,
        z: Nullable<float> = null,
        w: Nullable<float> = null,
    ): Vec4 {
        if (typeof x === "object") {
            w = x.w;
            z = x.z;
            y = x.y;
            x = x.x;
        } else if (y === null || z === null || w === null) {
            w = x;
            z = x;
            y = x;
        }
        this.x = x;
        this.y = y;
        this.z = z;
        this.w = w;
        return this;
    }

    public add(
        x: Vec4Like | float,
        y: Nullable<float> = null,
        z: Nullable<float> = null,
        w: Nullable<float> = null,
    ): Vec4 {
        if (typeof x === "object") {
            w = x.w;
            z = x.z;
            y = x.y;
            x = x.x;
        } else if (y === null || z === null || w === null) {
            w = x;
            z = x;
            y = x;
        }
        return this.set(this.x + x, this.y + y, this.z + z, this.w + w);
    }

    public sub(
        x: Vec4Like | float,
        y: Nullable<float> = null,
        z: Nullable<float> = null,
        w: Nullable<float> = null,
    ): Vec4 {
        if (typeof x === "object") {
            w = x.w;
            z = x.z;
            y = x.y;
            x = x.x;
        } else if (y === null || z === null || w === null) {
            w = x;
            z = x;
            y = x;
        }
        return this.set(this.x - x, this.y - y, this.z - z, this.w - w);
    }

    public scale(
        x: Vec4Like | float,
        y: Nullable<float> = null,
        z: Nullable<float> = null,
        w: Nullable<float> = null,
    ): Vec4 {
        if (typeof x === "object") {
            w = x.w;
            z = x.z;
            y = x.y;
            x = x.x;
        } else if (y === null || z === null || w === null) {
            w = x;
            z = x;
            y = x;
        }
        return this.set(this.x * x, this.y * y, this.z * z, this.w * w);
    }

    public divide(
        x: Vec4Like | float,
        y: Nullable<float> = null,
        z: Nullable<float> = null,
        w: Nullable<float> = null,
    ): Vec4 {
        if (typeof x === "object") {
            w = x.w;
            z = x.z;
            y = x.y;
            x = x.x;
        } else if (y === null || z === null || w === null) {
            w = x;
            z = x;
            y = x;
        }
        if (x === 0 || y === 0 || z === 0 || w === 0) {
            throw new Error("Vec4: Divide by zero.");
        }
        return this.set(this.x / x, this.y / y, this.z / z, this.w / w);
    }

    public lengthQuadratic(): float {
        return (
            this.x * this.x +
            this.y * this.y +
            this.z * this.z +
            this.w * this.w
        );
    }

    public length(): float {
        return Math.sqrt(this.lengthQuadratic());
    }

    public normalize(): Vec4 {
        return this.divide(this.length());
    }

    public dot(
        x: Vec4Like | float,
        y: Nullable<float> = null,
        z: Nullable<float> = null,
        w: Nullable<float> = null,
    ): float {
        if (typeof x === "object") {
            w = x.w;
            z = x.z;
            y = x.y;
            x = x.x;
        } else if (y === null || z === null || w === null) {
            w = x;
            z = x;
            y = x;
        }
        return this.x * x + this.y * y + this.z * z + this.w * w;
    }

    public multiply(matrix: Mat4): Vec4 {
        const v: Float32Array = matrix.values;
        return this.set(
            v[0] * this.x + v[1] * this.y + v[2] * this.z + v[3] * this.w,
            v[4] * this.x + v[5] * this.y + v[6] * this.z + v[7] * this.w,
            v[8] * this.x + v[9] * this.y + v[10] * this.z + v[11] * this.w,
            v[12] * this.x + v[13] * this.y + v[14] * this.z + v[15] * this.w,
        );
    }

    public copy(v: Vec4Like): Vec4 {
        return this.set(v.x, v.y, v.z, v.w);
    }

    public store(target: FloatArray, offset: int = 0): Vec4 {
        target[offset + 0] = this.x;
        target[offset + 1] = this.y;
        target[offset + 2] = this.z;
        target[offset + 3] = this.w;
        return this;
    }

    public clone(): Vec4 {
        return new Vec4(this);
    }

    public toJSON(): Object {
        return {
            x: this.x,
            y: this.y,
            z: this.z,
            w: this.w,
        };
    }

    public toArray(): [float, float, float, float] {
        return [this.x, this.y, this.z, this.w];
    }

    public toString(): string {
        return JSON.stringify(this);
    }

    public static Dot(a: Vec4, b: Vec4Like): float {
        return a.dot(b);
    }

    public static From(v: Vec2Like | Vec3Like | Vec4Like): Vec4 {
        if ("w" in v) {
            return new Vec4(v.x, v.y, v.z, v.w);
        }
        if ("z" in v) {
            return new Vec4(v.x, v.y, v.z);
        }
        return new Vec4(v.x, v.y);
    }
}
