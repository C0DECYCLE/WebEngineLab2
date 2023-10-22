struct Uniforms {
    matrix: mat4x4f,
    mode: u32, // 0: triangle, 1: cluster, 2: normal
};

struct Vertex {
    position: vec3f,
    clusterId: u32,
};

struct VertexShaderOut {
    @builtin(position) position: vec4f,
    @location(0) worldPosition: vec3f,
    @interpolate(flat) @location(1) flatWorldPosition: vec3f,
    @interpolate(flat) @location(2) clusterId: u32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> vertecies: array<Vertex>;

@vertex fn vs(@builtin(vertex_index) vertexIndex: u32) -> VertexShaderOut {
    let vertex: Vertex = vertecies[vertexIndex];
    var out: VertexShaderOut;
    out.position = uniforms.matrix * vec4f(vertex.position, 1.0);
    out.worldPosition = vertex.position;
    out.flatWorldPosition = vertex.position;
    out.clusterId = vertex.clusterId;
    return out;
}

@fragment fn fs(in: VertexShaderOut) -> @location(0) vec4f {
    if (uniforms.mode == 0) {
        return triangleShade(in.flatWorldPosition);
    }
    if (uniforms.mode == 1) {
        return clusterShade(in.clusterId);
    }
    if (uniforms.mode == 2) {
        return normalShade(in.worldPosition);
    }
    return vec4f(0.0, 0.0, 0.0, 1.0);
}

fn triangleShade(pos: vec3f) -> vec4f {
    let s: f32 = 100.0;
    let ox: f32 = pos.x * s;
    let oy: f32 = pos.y * s;
    let oz: f32 = pos.z * s;
    let r: f32 = noise3(vec3f(oy, oz, ox));
    let g: f32 = noise3(vec3f(oz, ox, oy));
    let b: f32 = noise3(vec3f(ox, oy, oz));
    return vec4f(r, g, b, 1.0);
}

fn clusterShade(id: u32) -> vec4f {
    if (id == 0) {
        return vec4f(1.0, 1.0, 1.0, 1.0);
    }
    let f: f32 = f32(id);
    let ox: f32 = f * 1.0;
    let oy: f32 = f * 2.0;
    let oz: f32 = f * 3.0;
    let r: f32 = noise3(vec3f(oy, oz, ox));
    let g: f32 = noise3(vec3f(oz, ox, oy));
    let b: f32 = noise3(vec3f(ox, oy, oz));
    return vec4f(r, g, b, 1.0);
}

fn normalShade(pos: vec3f) -> vec4f {
    let normal: vec3f = normalize(cross(dpdx(pos), dpdy(pos)));
    return vec4f(normal * 0.5 + 0.5, 1.0);
}

fn mod289(x: vec4f) -> vec4f {
    return x - floor(x * (1. / 289.)) * 289.;
}

fn perm4(x: vec4f) -> vec4f {
    return mod289(((x * 34.) + 1.) * x);
}

fn noise3(p: vec3f) -> f32 {
    let a: vec3f = floor(p);
    var d: vec3f = p - a;
    d = d * d * (3. - 2. * d);
    let b: vec4f = a.xxyy + vec4f(0., 1., 0., 1.);
    let k1: vec4f = perm4(b.xyxy);
    let k2: vec4f = perm4(k1.xyxy + b.zzww);
    let c: vec4f = k2 + a.zzzz;
    let k3: vec4f = perm4(c);
    let k4: vec4f = perm4(c + 1.);
    let o1: vec4f = fract(k3 * (1. / 41.));
    let o2: vec4f = fract(k4 * (1. / 41.));
    let o3: vec4f = o2 * d.z + o1 * (1. - d.z);
    let o4: vec2f = o3.yw * d.x + o3.xz * (1. - d.x);
    return o4.y * d.y + o4.x * (1. - d.y);
}
