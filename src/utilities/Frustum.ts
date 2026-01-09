/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger
 */

import { float } from "./utils.type.js";
import { Mat4 } from "./Mat4.js";
import { Vec3 } from "./Vec3.js";

export type Plane = {
    readonly normal: Vec3;
    readonly distance: float;
};

export class Frustum {
    public readonly left: Plane;
    public readonly right: Plane;
    public readonly top: Plane;
    public readonly bottom: Plane;
    public readonly near: Plane;
    public readonly far: Plane;

    public constructor(viewProjection: Mat4) {
        this.left = this.extractLeft(viewProjection);
        this.right = this.extractRight(viewProjection);
        this.top = this.extractTop(viewProjection);
        this.bottom = this.extractBottom(viewProjection);
        this.near = this.extractNear(viewProjection);
        this.far = this.extractFar(viewProjection);
    }

    private extractLeft(viewProjection: Mat4): Plane {
        return this.extractPlane(
            viewProjection.values[3] + viewProjection.values[0],
            viewProjection.values[7] + viewProjection.values[4],
            viewProjection.values[11] + viewProjection.values[8],
            viewProjection.values[15] + viewProjection.values[12],
        );
    }

    private extractRight(viewProjection: Mat4): Plane {
        return this.extractPlane(
            viewProjection.values[3] - viewProjection.values[0],
            viewProjection.values[7] - viewProjection.values[4],
            viewProjection.values[11] - viewProjection.values[8],
            viewProjection.values[15] - viewProjection.values[12],
        );
    }

    private extractTop(viewProjection: Mat4): Plane {
        return this.extractPlane(
            viewProjection.values[3] - viewProjection.values[1],
            viewProjection.values[7] - viewProjection.values[5],
            viewProjection.values[11] - viewProjection.values[9],
            viewProjection.values[15] - viewProjection.values[13],
        );
    }

    private extractBottom(viewProjection: Mat4): Plane {
        return this.extractPlane(
            viewProjection.values[3] + viewProjection.values[1],
            viewProjection.values[7] + viewProjection.values[5],
            viewProjection.values[11] + viewProjection.values[9],
            viewProjection.values[15] + viewProjection.values[13],
        );
    }

    private extractNear(viewProjection: Mat4): Plane {
        return this.extractPlane(
            viewProjection.values[2],
            viewProjection.values[6],
            viewProjection.values[10],
            viewProjection.values[14],
        );
    }

    private extractFar(viewProjection: Mat4): Plane {
        return this.extractPlane(
            viewProjection.values[3] - viewProjection.values[2],
            viewProjection.values[7] - viewProjection.values[6],
            viewProjection.values[11] - viewProjection.values[10],
            viewProjection.values[15] - viewProjection.values[14],
        );
    }

    private extractPlane(a: float, b: float, c: float, d: float): Plane {
        const normal: Vec3 = new Vec3(a, b, c);
        const length: float = normal.length();
        return { normal: normal.divide(length), distance: d / length } as Plane;
    }
}
