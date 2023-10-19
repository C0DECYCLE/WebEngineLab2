/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, October 2023
 */

import { int, float } from "./../../types/utils.js";

export class Vector2 {
    constructor(public x: int | float, public y: int | float) {}

    add(value: Vector2): Vector2 {
        return new Vector2(this.x + value.x, this.y + value.y);
    }
}

export class Vector3 {
    constructor(
        public x: int | float,
        public y: int | float,
        public z: int | float,
    ) {}

    add(value: Vector3): Vector3 {
        return new Vector3(
            this.x + value.x,
            this.y + value.y,
            this.z + value.z,
        );
    }
}
