/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, October 2023
 */

import { log } from "./utilities/logger.js";
import { Vector2, Vector3 } from "./utilities/vectors.js";

const vec2a: Vector2 = new Vector2(1, 2);
const vec2b: Vector2 = new Vector2(2, 1);
log(vec2a.add(vec2b));

const vec3a: Vector3 = new Vector3(1, 2, 3);
const vec3b: Vector3 = new Vector3(3, 2, 1);
log(vec3a.add(vec3b));
