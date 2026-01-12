struct Dimensions {
    width: u32,
    height: u32,
    generation_offset: u32,
    num_of_particles: u32,
    frame_time: f32,
    is_gravity_on: u32,
    time_to_die: f32,
    num_of_particles_to_generate_per_second: u32,
    target_pos: vec4<f32>,
    proj_view: mat4x4<f32>,
    init_type: u32,
}

struct particle {
    pos: vec4<f32>,
    speed: vec4<f32>,
    accel: vec4<f32>,
}

struct number_of_alive_particles {
    count: atomic<u32>,
}

var<workgroup> shared_reduce: array<u32, 256>;

@group(0) @binding(1)
var<storage, read_write> particles: array<particle>;

@group(0) @binding(2)
var<uniform> dimensions: Dimensions;

@group(0) @binding(3)
var<storage, read_write> alive_particles: number_of_alive_particles;

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
    // let e2 = fma(- a, x, 1.0);
    // x = fma(x, e2, x);

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

const INT_max_u32 = 1.0 / 4294967295.0;

fn randomFloat(value: u32) -> f32 {
    return f32(hash(value)) * INT_max_u32;
}

fn inverseSqrt(value: f32) -> f32 {
    return fast_reciprocal_fma(sqrt(value));
}

const PI: f32 = 3.14159265359;
const TWO_PI: f32 = 6.28318530718;
const ONE_DIV_3: f32 = 1.0 / 3.0;

fn get_cube_pos(id: u32) -> vec4<f32> {
     return vec4<f32>(fma(randomFloat(id), 50.0, - 25.0), fma(randomFloat(id * 2u), 50.0, - 25.0), fma(randomFloat(id * 3u), - 50.0, 25.0), 0.0);
}

fn get_sphere_pos(id: u32) -> vec4<f32> {
    // Spherical coordinates method for uniform distribution
    let u = randomFloat(id) * 2.0 - 1.0;
    let theta = randomFloat(id * 2u) * TWO_PI; // 2*PI

    // Cube root of random variable ensures uniform distribution in volume
    let random_radius_scale = pow(randomFloat(id * 3u), ONE_DIV_3);
    let r = 40.0 * random_radius_scale;

    let r_xy = r * sqrt(1.0 - u * u);

    return vec4<f32>(
        r_xy * cos(theta),
        r_xy * sin(theta),
        r * u,
        0.0);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {

    // Parallel Reduction: Step 1 - Load into shared memory
    var local_val = 0u;
    if (global_id.x < dimensions.num_of_particles) {
        particles[global_id.x].pos.w = 0.0;
        local_val = select(0u, 1u, particles[global_id.x].accel.w >= 1.0);
    }
    shared_reduce[local_id.x] = local_val;
    workgroupBarrier();

    // Parallel Reduction: Step 2 - Reduce in shared memory
    // Workgroup size is 256, so we start stride at 128
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (local_id.x < s) {
            shared_reduce[local_id.x] += shared_reduce[local_id.x + s];
        }
        workgroupBarrier();
    }

    // Parallel Reduction: Step 3 - Add partial sum to global counter
    if (local_id.x == 0u) {
        atomicAdd(&alive_particles.count, shared_reduce[0]);
    }


    var p = particles[global_id.x];

    let start_gen = dimensions.generation_offset * dimensions.num_of_particles_to_generate_per_second;
    let end_gen = start_gen + dimensions.num_of_particles_to_generate_per_second;
    let should_revive = (p.accel.w <= 0.0) && (global_id.x >= start_gen) && (global_id.x < end_gen);

    if (should_revive) {
        p.accel.w = 1.0;
        p.speed.w = 0.0;
        if (dimensions.init_type == 0u) {
             p.pos = get_cube_pos(global_id.x);
        } else {
             p.pos = get_sphere_pos(global_id.x);
        }
    }
    let dt = vec3<f32>(dimensions.frame_time);
    p.speed.w = p.speed.w + dt.x;
    p.accel.w = select(p.accel.w, 0.0, (p.speed.w >= dimensions.time_to_die) && (dimensions.time_to_die > 0.0));


    // Optimization 2: Fused Multiply-Add (FMA) for Physics
    // Executes a * b + c in a single instruction cycle
    p.speed = vec4<f32>(fma(p.accel.xyz, dt, p.speed.xyz), p.speed.w);
    p.pos = vec4<f32>(fma(p.speed.xyz, dt, p.pos.xyz), p.pos.w);

    // Frame-rate independent damping (approx 0.994 at 60 FPS)
    // Uses FMA: speed = speed * (-0.4 * dt) + speed  =>  speed * (1.0 - 0.4 * dt)
    p.speed = vec4<f32>(fma(p.speed.xyz, -dt * 0.7, p.speed.xyz), p.speed.w);



    if( dimensions.is_gravity_on == 0u) {
        p.accel =  vec4<f32>(0.0, 0.0, 0.0, p.accel.w);
        particles[global_id.x] = p;
        return;
    }
    let target_pos = dimensions.target_pos.xyz;
    let diff = target_pos - p.pos.xyz;
    let dist_sq = dot(diff, diff);

    let r_inv = inverseSqrt(max(dist_sq, 0.001));
    // Fast hardware instruction
    let force_mag = 30000.0 * fast_reciprocal_fma(dist_sq + 100.0);

    // Result direction is normalized implicitly here by multiplication
    p.accel = vec4<f32>(diff * (r_inv * force_mag), p.accel.w);

    // Write back to global memory once
    particles[global_id.x] = p;
}

@compute @workgroup_size(256)
fn init_cube(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= dimensions.num_of_particles) {
        return;
    }
    var dead_or_alive = select(0.0, 1.0, dimensions.time_to_die <= 0.0);
    dead_or_alive = select(dead_or_alive, 1.0, particles[global_id.x].accel.w >= 1.0);

    particles[global_id.x] = particle(
        get_cube_pos(global_id.x), // pos
        vec4<f32>(0.0, 0.0, 0.0, particles[global_id.x].speed.w), /* speed w is time spent alive to date */
        vec4<f32>(0.0, 0.0, 0.0,  dead_or_alive), /* accel 0.0 if dead, 1.0 if alive */);
}

@compute @workgroup_size(256)
fn init_sphere(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= dimensions.num_of_particles) {
        return;
    }

    var dead_or_alive = select(0.0, 1.0, dimensions.time_to_die <= 0.0);
    dead_or_alive = select(dead_or_alive, 1.0, particles[global_id.x].accel.w >= 1.0);
    particles[global_id.x] = particle(
        get_sphere_pos(global_id.x), // pos
        vec4<f32>(0.0, 0.0, 0.0, particles[global_id.x].speed.w), /* speed */
        vec4<f32>(0.0, 0.0, 0.0, dead_or_alive ), /* accel */);
}
