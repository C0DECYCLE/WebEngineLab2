/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, October 2023
 */

import { int } from "../../types/utilities/utils.type.js";

export const byteSize: int = 4;

export function createCanvas(): HTMLCanvasElement {
    const canvas: HTMLCanvasElement = document.createElement("canvas");
    canvas.width = document.body.clientWidth * devicePixelRatio;
    canvas.height = document.body.clientHeight * devicePixelRatio;
    canvas.style.position = "absolute";
    canvas.style.top = "0px";
    canvas.style.left = "0px";
    canvas.style.width = "100%";
    canvas.style.height = "100%";
    document.body.appendChild(canvas);
    return canvas;
}
