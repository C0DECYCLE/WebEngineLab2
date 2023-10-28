/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, October 2023
 */

import { float, MapString } from "../types/utilities/utils.type.js";
import { toRadian, clamp } from "./utilities/utils.js";
import { Vec3 } from "./utilities/Vec3.js";
import { Mat4 } from "./utilities/Mat4.js";

export class Controller {
    private readonly camera: { position: Vec3; direction: Vec3 };
    private readonly pressing: MapString<boolean> = new MapString<boolean>();
    private readonly globalUp: Vec3 = new Vec3(0.0, 1.0, 0.0);

    private direction: Vec3 = new Vec3();
    private left: Vec3 = new Vec3();
    private localUp: Vec3 = new Vec3(0.0, 1.0, 0.0);
    private transform: Mat4 = new Mat4();

    private velocity: float = 0.35;
    private locked: boolean = false;

    public constructor(camera: { position: Vec3; direction: Vec3 }) {
        this.camera = camera;
        this.register();
    }

    public update(): void {
        this.keyboardPosition();
    }

    private register(): void {
        document.addEventListener("keydown", (event: KeyboardEvent) => {
            event.preventDefault();
            this.pressing.set(event.key.toLowerCase(), true);
        });
        document.addEventListener("keyup", (event: KeyboardEvent) => {
            event.preventDefault();
            this.pressing.set(event.key.toLowerCase(), false);
        });
        document.addEventListener("wheel", (event: WheelEvent) => {
            this.velocity -= event.deltaY * 0.0001;
            this.velocity = clamp(this.velocity, 0.01, 1.0);
        });
        document.addEventListener("pointerlockchange", (_event: Event) => {
            this.locked = !this.locked;
        });
        document.addEventListener("mousemove", (event: MouseEvent) => {
            if (!this.locked) {
                return;
            }
            this.mouseDirection(event.movementX, event.movementY);
        });
    }

    private keyboardPosition(): void {
        this.direction.set(0.0, 0.0, 0.0);
        this.left.copy(this.camera.direction).cross(this.globalUp);
        this.localUp.copy(this.left).cross(this.camera.direction);
        if (this.pressing.get("w") === true) {
            this.direction.sub(this.camera.direction);
        } else if (this.pressing.get("s") === true) {
            this.direction.add(this.camera.direction);
        }
        if (this.pressing.get("a") === true) {
            this.direction.add(this.left);
        } else if (this.pressing.get("d") === true) {
            this.direction.sub(this.left);
        }
        if (this.pressing.get(" ") === true) {
            this.direction.add(this.localUp);
        } else if (this.pressing.get("control") === true) {
            this.direction.sub(this.localUp);
        }
        if (!this.direction.x && !this.direction.y && !this.direction.z) {
            return;
        }
        this.camera.position.add(
            this.direction.normalize().scale(this.velocity),
        );
    }

    private mouseDirection(x: float, y: float): void {
        this.transform.reset();
        this.transform.rotateY(-x * 0.1 * toRadian);
        this.camera.direction.applyMat(this.transform);
        this.left.copy(this.camera.direction).cross(this.globalUp);
        this.transform.reset();
        this.transform.rotateAxis(this.left.normalize(), y * 0.1 * toRadian);
        this.camera.direction.applyMat(this.transform).normalize();
    }
}
