/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger
 */

import { int, float, Nullable, FloatArray } from "./utils.type.js";
import { Mat3 } from "./Mat3.js";
import { Vec2Like } from "./Vec2.js";
import { Vec4Like } from "./Vec4.js";

export type Vec3Like = {
    x: float;
    y: float;
    z: float;
};

export class Vec3 {
    public static Cache: Vec3 = new Vec3();

    protected _x!: float;
    protected _y!: float;
    protected _z!: float;

    public get x(): float {
        return this._x;
    }

    public get y(): float {
        return this._y;
    }

    public get z(): float {
        return this._z;
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

    public constructor(x: Vec3Like | float = 0, y: float = 0, z: float = 0) {
        this.set(x, y, z);
    }

    public set(
        x: Vec3Like | float,
        y: Nullable<float> = null,
        z: Nullable<float> = null,
    ): Vec3 {
        if (typeof x === "object") {
            z = x.z;
            y = x.y;
            x = x.x;
        } else if (y === null || z === null) {
            z = x;
            y = x;
        }
        this.x = x;
        this.y = y;
        this.z = z;
        return this;
    }

    public add(
        x: Vec3Like | float,
        y: Nullable<float> = null,
        z: Nullable<float> = null,
    ): Vec3 {
        if (typeof x === "object") {
            z = x.z;
            y = x.y;
            x = x.x;
        } else if (y === null || z === null) {
            z = x;
            y = x;
        }
        return this.set(this.x + x, this.y + y, this.z + z);
    }

    public sub(
        x: Vec3Like | float,
        y: Nullable<float> = null,
        z: Nullable<float> = null,
    ): Vec3 {
        if (typeof x === "object") {
            z = x.z;
            y = x.y;
            x = x.x;
        } else if (y === null || z === null) {
            z = x;
            y = x;
        }
        return this.set(this.x - x, this.y - y, this.z - z);
    }

    public scale(
        x: Vec3Like | float,
        y: Nullable<float> = null,
        z: Nullable<float> = null,
    ): Vec3 {
        if (typeof x === "object") {
            z = x.z;
            y = x.y;
            x = x.x;
        } else if (y === null || z === null) {
            z = x;
            y = x;
        }
        return this.set(this.x * x, this.y * y, this.z * z);
    }

    public divide(
        x: Vec3Like | float,
        y: Nullable<float> = null,
        z: Nullable<float> = null,
    ): Vec3 {
        if (typeof x === "object") {
            z = x.z;
            y = x.y;
            x = x.x;
        } else if (y === null || z === null) {
            z = x;
            y = x;
        }
        if (x === 0 || y === 0 || z === 0) {
            throw new Error("Vec3: Divide by zero.");
        }
        return this.set(this.x / x, this.y / y, this.z / z);
    }

    public lengthQuadratic(): float {
        return this.x * this.x + this.y * this.y + this.z * this.z;
    }

    public length(): float {
        return Math.sqrt(this.lengthQuadratic());
    }

    public normalize(): Vec3 {
        return this.divide(this.length());
    }

    public dot(
        x: Vec3Like | float,
        y: Nullable<float> = null,
        z: Nullable<float> = null,
    ): float {
        if (typeof x === "object") {
            z = x.z;
            y = x.y;
            x = x.x;
        } else if (y === null || z === null) {
            z = x;
            y = x;
        }
        return this.x * x + this.y * y + this.z * z;
    }

    public cross(b: Vec3Like, a: Vec3Like = this): Vec3 {
        return this.set(
            a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x,
        );
    }

    public multiply(matrix: Mat3): Vec3 {
        const v: Float32Array = matrix.values;
        return this.set(
            v[0] * this.x + v[1] * this.y + v[2] * this.z,
            v[3] * this.x + v[4] * this.y + v[5] * this.z,
            v[6] * this.x + v[7] * this.y + v[8] * this.z,
        );
    }

    public copy(v: Vec3Like): Vec3 {
        return this.set(v.x, v.y, v.z);
    }

    public store(target: FloatArray, offset: int = 0): Vec3 {
        target[offset + 0] = this.x;
        target[offset + 1] = this.y;
        target[offset + 2] = this.z;
        return this;
    }

    public clone(): Vec3 {
        return new Vec3(this);
    }

    public toJSON(): Object {
        return {
            x: this.x,
            y: this.y,
            z: this.z,
        };
    }

    public toArray(): [float, float, float] {
        return [this.x, this.y, this.z];
    }

    public toString(): string {
        return JSON.stringify(this);
    }

    public static Dot(a: Vec3, b: Vec3Like): float {
        return a.dot(b);
    }

    public static Cross(a: Vec3, b: Vec3Like): Vec3 {
        return a.clone().cross(b);
    }

    public static From(v: Vec2Like | Vec3Like | Vec4Like): Vec3 {
        if ("z" in v) {
            return new Vec3(v.x, v.y, v.z);
        }
        return new Vec3(v.x, v.y);
    }
}
