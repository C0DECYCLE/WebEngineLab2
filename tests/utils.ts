import { int, float } from "../types/utilities/utils.type.js";
import { Vec3 } from "../src/utilities/Vec3.js";

export function expectFloatArrayToBeCloseTo(
    received: float[] | Float32Array | Float64Array,
    expected: float[] | Float32Array | Float64Array,
): void {
    expect(received.length).toBe(expected.length);
    received.forEach((x: float, i: int) => expect(x).toBeCloseTo(expected[i]));
}

export function expectVec3ToBeCloseTo(
    received: Vec3,
    expectedX: float,
    expectedY: float,
    expectedZ: float,
): void {
    expect(received.x).toBeCloseTo(expectedX);
    expect(received.y).toBeCloseTo(expectedY);
    expect(received.z).toBeCloseTo(expectedZ);
}
