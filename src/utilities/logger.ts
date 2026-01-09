/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger
 */

import { int } from "./utils.type.js";

export function log(...data: any[]): void {
    return logger("log", ...data);
}

export function warn(...data: any[]): void {
    return logger("warn", ...data);
}

export const logger_archive: string[] = [];
const logger_maximum: int = 512;
let logger_count: int = 0;
let logger_exit: boolean = false;

function logger(type: "log" | "warn", ...data: any[]): void {
    if (logger_exit) {
        return;
    }
    if (logger_count === logger_maximum) {
        logger_exit = true;
        return console.warn("Logger: Maximum count exceeded.");
    }
    logger_count++;
    logger_archive.push(data.toString());
    console[type](...data);
}
