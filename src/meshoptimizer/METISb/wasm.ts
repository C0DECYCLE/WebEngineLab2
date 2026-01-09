// @ts-nocheck

// DO NOT ADD IMPORTS TO '*.d.ts' files!
// https://stackoverflow.com/a/51114250

type TypedArray = Float32Array | Uint8Array | Uint32Array;

type WasmPtrInout = "in" | "out";

/** This struct is only needed to know we need to copy back data from wasm's heap.
 * https://gamedev.stackexchange.com/questions/29672/in-out-keywords-in-glsl
 */
type WasmPtr = {
    arr: WasmArray;
    inout: WasmPtrInout;
};

type WasmArray = Float32Array | Uint32Array | Int32Array | Uint8Array;
type WasmHeap = "HEAPF32" | "HEAPU32" | "HEAP32" | "HEAPU8";
type WasmBasicTypeName = "number" | "string" | "array";
type WasmBasicType = number | string | WasmPtr | null;

type WasmHeapAdr = number;

type WasmModuleWithHeaps = Record<WasmHeap, TypedArray>;

// Monkey patch with emscripten
// declare namespace WebAssembly {
interface WasmModule extends WasmModuleWithHeaps, WebAssembly.Module {
    ccall(
        fnName: string,
        returnTypeName: WasmBasicTypeName,
        argsTypes: string[],
        argsWasm: WasmBasicType[],
        // deno-lint-ignore no-explicit-any
    ): any;
    _malloc(bytesCnt: number): WasmHeapAdr;
    _create_buffer(bytesCnt: number): WasmHeapAdr;
}
// }

export function getClassName(a: object) {
    // deno-lint-ignore no-explicit-any
    return (a as any).constructor.name;
}

export function getTypeName(a: any) {
    if (Array.isArray(a)) return "Array";
    if (typeof a === "object") return getClassName(a);
    return typeof a;
}

// Some references:
// - https://marcoselvatici.github.io/WASM_tutorial/
// - https://radu-matei.com/blog/practical-guide-to-wasm-memory/

const DEBUG_ALLOC = false;
// TODO [LOW] re-Emscripten metis with free(). ATM we use metisFreeAllocations()
// to mass free() the memory at the end of object processing.
// Metis does not allocate much memory, so you will not alloc >2GB.
// Meshoptimizer on the other hand... each I've exported _free() and hope it works.
const SILENCE_MEMORY_LEAKS = true;

const heapPtr2str = <T extends WasmArray>(
    module: WasmModule,
    arr: T,
    ptr: number,
) => {
    const moduleName = 'metis'; // prettier-ignore
    return `${moduleName}@0x${ptr.toString(16)}: ${arr.byteLength} bytes (${getTypeName(arr)}: ${arr.length} items)`; // prettier-ignore
};

// https://stackoverflow.com/questions/41875728/pass-a-javascript-array-as-argument-to-a-webassembly-function
export const wasmPtr = (arr: WasmArray, inout: WasmPtrInout = "in") => ({
    arr,
    inout,
});

function copyToWasmHeap<T extends WasmArray>(
    module: WasmModule,
    heap: WasmHeap,
    arr: T,
) {
    const mallocFn = "_create_buffer" in module ? "_create_buffer" : "_malloc";
    const heapPointer = module[mallocFn](arr.length * arr.BYTES_PER_ELEMENT);
    module[heap].set(arr, heapPointer / arr.BYTES_PER_ELEMENT);
    if (DEBUG_ALLOC) {
        console.log(`Alloc ${heapPtr2str(module, arr, heapPointer)}`);
    }
    return heapPointer;
}

function getMemoryHeapForJsArray(arr: WasmArray): WasmHeap | undefined {
    if (arr instanceof Float32Array) return "HEAPF32";
    if (arr instanceof Uint32Array) return "HEAPU32";
    if (arr instanceof Int32Array) return "HEAP32";
    if (arr instanceof Uint8Array) return "HEAPU8";
    return undefined;
}

function transformIntoWasmArg(
    module: WasmModule,
    fnName: string,
    arg: WasmBasicType,
) {
    if (typeof arg === "number" || typeof arg === "string" || arg === null) {
        return arg;
    }
    const heap = getMemoryHeapForJsArray(arg.arr);
    if (heap) {
        return copyToWasmHeap(module, heap, arg.arr);
    }

    const s = JSON.stringify(arg);
    throw new Error(
        `Wasm function '${fnName}' received invalid argument: ${s}`,
    );
}

export const wasmCall = <RetType>(
    module: WasmModule,
    returnTypeName: WasmBasicTypeName,
    fnName: string,
    argsJS: WasmBasicType[],
): RetType => {
    const argsWasm = argsJS.map((arg) =>
        transformIntoWasmArg(module, fnName, arg),
    );
    const argsTypes = argsJS.map(
        (arg): WasmBasicTypeName =>
            typeof arg === "string" ? "string" : "number",
    );

    let result: RetType;
    if ("ccall" in module) {
        result = module.ccall(fnName, returnTypeName, argsTypes, argsWasm);
    } else {
        // deno-lint-ignore no-explicit-any
        result = (module as any)[`_${fnName}`](...argsWasm);
    }

    argsJS.forEach((arg, i) => {
        const rawHeapAddr = argsWasm[i];
        if (
            typeof arg === "number" ||
            typeof arg === "string" ||
            typeof rawHeapAddr === "string" ||
            arg === null
        ) {
            return;
        }

        // copy back from wasm's heap to JS TypedArray
        const heap = getMemoryHeapForJsArray(arg.arr)!;
        const wasmHeapAddr = rawHeapAddr! / arg.arr.BYTES_PER_ELEMENT; // convert to emscripten

        if (arg.inout === "out") {
            // console.log(`${fnName} HEAP`, module[heap]);
            const length = arg.arr.length;
            // copy into the user-provided array
            arg.arr.set(
                module[heap].subarray(wasmHeapAddr, wasmHeapAddr + length),
            );
        }

        // free the allocated heap memory
        if (DEBUG_ALLOC) {
            console.log(`[${fnName}] Will free  ${heapPtr2str(module, arg.arr, rawHeapAddr)}`); // prettier-ignore
        }
        if (module._free !== undefined) {
            module._free(rawHeapAddr);
        } else if (!SILENCE_MEMORY_LEAKS) {
            console.error(`Memory leak [${fnName}]: ${heapPtr2str(module, arg.arr, rawHeapAddr)}`, { module, fnName }); // prettier-ignore
        }
    });

    return result;
};

export const metisCall = <
    FnName extends keyof typeof metis,
    RetType extends ReturnType<(typeof metis)[FnName]>,
>(
    module: WasmModule,
    returnTypeName: WasmBasicTypeName,
    fnName: FnName,
    argsJS: Parameters<(typeof metis)[FnName]>,
): RetType => {
    // deno-lint-ignore no-explicit-any
    return wasmCall(module, returnTypeName, fnName as any, argsJS);
};
