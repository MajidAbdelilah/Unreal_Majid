struct Dimensions {
    width: u32,
    height: u32,
    stride: u32,
    num_of_particles: u32,
    frame_time: f32,
    _pad1: u32,
    _pad2: vec2<u32>,
    target_pos: vec4<f32>,
    proj_view: mat4x4<f32>,
}

struct particle {
    pos: vec4<f32>,
    speed: vec4<f32>,
    accel: vec4<f32>,
}

@group(0) @binding(1)
var<storage, read_write> particles: array<particle>;

@group(0) @binding(2)
var<uniform> dimensions: Dimensions;

fn fast_reciprocal_fma(a: f32) -> f32 {
    // 1. Initial guess (approx 7-8 bits precision)
    let i = bitcast<u32>(a);
    let i_approx = 0x7EEEEBB3u - i;
    var x = bitcast<f32>(i_approx);

    // 2. First Iteration (Refines to ~14 bits)
    // Calculate residual: e = 1.0 - (a * x)
    let e1 = fma(- a, x, 1.0);
    // Apply correction: x = x + (x * e1)
    x = fma(x, e1, x);

    // 3. Second Iteration (Refines to full f32 precision)
    let e2 = fma(- a, x, 1.0);
    x = fma(x, e2, x);

    return x;
}

fn hash(value: u32) -> u32 {
    var state = value;
    state = state ^ 2747636419u;
    state = state * 2654435769u;
    state = state ^ (state >> 16u);
    state = state * 2654435769u;
    state = state ^ (state >> 16u);
    state = state * 2654435769u;
    return state;
}

const INV_max_u32 = 1.0 / 4294967295.0;

fn randomFloat(value: u32) -> f32 {
    return f32(hash(value)) * INV_max_u32;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= dimensions.num_of_particles) {
        return;
    }

    // Optimization 1: Memory Coalescing (Load once into registers)
    var p = particles[global_id.x];
    let dt = vec4<f32>(dimensions.frame_time);

    // Optimization 2: Fused Multiply-Add (FMA) for Physics
    // Executes a * b + c in a single instruction cycle
    p.speed = fma(p.accel, dt, p.speed);
    p.pos = fma(p.speed, dt, p.pos);
    p.speed *= 0.99;

    let target_pos = dimensions.target_pos.xyz;
    let diff = target_pos - p.pos.xyz;
    let dist_sq = dot(diff, diff);

    let r_inv = inverseSqrt(max(dist_sq, 0.001));
    // Fast hardware instruction
    let force_mag = 100000.0 * fast_reciprocal_fma(dist_sq + 100.0);

    // Result direction is normalized implicitly here by multiplication
    p.accel = vec4<f32>(diff * (r_inv * force_mag), 0.0);

    // Write back to global memory once
    particles[global_id.x] = p;
}

@compute @workgroup_size(256)
fn init(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x > dimensions.num_of_particles) {
        return;
    }

    particles[global_id.x] = particle(vec4<f32>(fma(randomFloat(global_id.x), 50.0, - 25.0), fma(randomFloat(global_id.x * 2), 50.0, - 25.0), fma(randomFloat(global_id.x * 3), - 50.0, 25.0), 1.0), // pos
    vec4<f32>(0.0, 0.0, 0.0, 0.0), /* speed */

    vec4<f32>(0.0, 0.0, 0.0, 0.0) /* accel */);
}
