/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger
 */

import { int, float, Nullable, FloatArray } from "./utils.type.js";
import { Mat2 } from "./Mat2.js";
import { normalizeRadian } from "./utils.js";
import { Vec3Like } from "./Vec3.js";
import { Vec4Like } from "./Vec4.js";

export type Vec2Like = {
    x: float;
    y: float;
};

export class Vec2 {
    public static Cache: Vec2 = new Vec2();

    protected _x!: float;
    protected _y!: float;

    public get x(): float {
        return this._x;
    }

    public get y(): float {
        return this._y;
    }

    public set x(value: float) {
        this._x = value;
    }

    public set y(value: float) {
        this._y = value;
    }

    public constructor(x: Vec2Like | float = 0, y: float = 0) {
        this.set(x, y);
    }

    public set(x: Vec2Like | float, y: Nullable<float> = null): Vec2 {
        if (typeof x === "object") {
            y = x.y;
            x = x.x;
        } else if (y === null) {
            y = x;
        }
        this.x = x;
        this.y = y;
        return this;
    }

    public add(x: Vec2Like | float, y: Nullable<float> = null): Vec2 {
        if (typeof x === "object") {
            y = x.y;
            x = x.x;
        } else if (y === null) {
            y = x;
        }
        return this.set(this.x + x, this.y + y);
    }

    public sub(x: Vec2Like | float, y: Nullable<float> = null): Vec2 {
        if (typeof x === "object") {
            y = x.y;
            x = x.x;
        } else if (y === null) {
            y = x;
        }
        return this.set(this.x - x, this.y - y);
    }

    public scale(x: Vec2Like | float, y: Nullable<float> = null): Vec2 {
        if (typeof x === "object") {
            y = x.y;
            x = x.x;
        } else if (y === null) {
            y = x;
        }
        return this.set(this.x * x, this.y * y);
    }

    public divide(x: Vec2Like | float, y: Nullable<float> = null): Vec2 {
        if (typeof x === "object") {
            y = x.y;
            x = x.x;
        } else if (y === null) {
            y = x;
        }
        if (x === 0 || y === 0) {
            throw new Error("Vec2: Divide by zero.");
        }
        return this.set(this.x / x, this.y / y);
    }

    public lengthQuadratic(): float {
        return this.x * this.x + this.y * this.y;
    }

    public length(): float {
        return Math.sqrt(this.lengthQuadratic());
    }

    public normalize(): Vec2 {
        return this.divide(this.length());
    }

    public dot(x: Vec2Like | float, y: Nullable<float> = null): float {
        if (typeof x === "object") {
            y = x.y;
            x = x.x;
        } else if (y === null) {
            y = x;
        }
        return this.x * x + this.y * y;
    }

    public rotate(radian: float): Vec2 {
        radian = normalizeRadian(radian);
        const c: float = Math.cos(radian);
        const s: float = Math.sin(radian);
        // prettier-ignore
        return this.set(
            c * this.x - s * this.y,
            s * this.x + c * this.y
        );
    }

    public multiply(matrix: Mat2): Vec2 {
        const v: Float32Array = matrix.values;
        return this.set(
            v[0] * this.x + v[1] * this.y,
            v[2] * this.x + v[3] * this.y,
        );
    }

    public copy(v: Vec2Like): Vec2 {
        return this.set(v.x, v.y);
    }

    public store(target: FloatArray, offset: int = 0): Vec2 {
        target[offset + 0] = this.x;
        target[offset + 1] = this.y;
        return this;
    }

    public clone(): Vec2 {
        return new Vec2(this);
    }

    public toJSON(): Object {
        return {
            x: this.x,
            y: this.y,
        };
    }

    public toArray(): [float, float] {
        return [this.x, this.y];
    }

    public toString(): string {
        return JSON.stringify(this);
    }

    public static Dot(a: Vec2, b: Vec2Like): float {
        return a.dot(b);
    }

    public static From(v: Vec2Like | Vec3Like | Vec4Like): Vec2 {
        return new Vec2(v.x, v.y);
    }
}
