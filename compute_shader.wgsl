struct Dimensions {
    width: u32,
    height: u32,
    stride: u32,
    num_of_particles: u32,
    frame_time: f32,
    _pad1: u32,
    _pad2: vec2<u32>,
    target_pos: vec4<f32>,
    proj: mat4x4<f32>,
    view: mat4x4<f32>,
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

fn hash(value: u32) -> u32 {
    var state = value;
    state = state ^ 2747636419u;
    state = state * 2654435769u;
    state = state ^ state >> 16u;
    state = state * 2654435769u;
    state = state ^ state >> 16u;
    state = state * 2654435769u;
    return state;
}

fn randomFloat(value: u32) -> f32 {
    return f32(hash(value)) / 4294967295.0;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let coords = global_id.xy;

    if (global_id.x > dimensions.num_of_particles) {
        return;
    }

    particles[global_id.x].speed += particles[global_id.x].accel * dimensions.frame_time;
    particles[global_id.x].pos += particles[global_id.x].speed * dimensions.frame_time;
    particles[global_id.x].speed *= 0.99;

    let target_pos = dimensions.target_pos.xyz;

    let diff = target_pos - particles[global_id.x].pos.xyz;
    let dist_sq = dot(diff, diff);
    let dir = normalize(diff);

    // Attraction force (Gravity-like: F = G / r^2)
    // Added epsilon (100.0) to prevent division by zero and extreme forces
    let force_mag = 100000.0 / (dist_sq + 100.0);

    particles[global_id.x].accel = vec4<f32>(dir * force_mag, 0.0);
}

@compute @workgroup_size(256)
fn init(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x > dimensions.num_of_particles) {
        return;
    }

    particles[global_id.x] = particle(vec4<f32>(randomFloat(global_id.x) * 50.0 - 25.0, randomFloat(global_id.x * 2) * 50.0 - 25.0, randomFloat(global_id.x * 3) * - 50.0 + 25.0, 1.0), // pos
    // vec4<f32>(randomFloat(global_id.x) * 20.0 - 10.0, randomFloat(global_id.x * 2) * 20.0 - 10.0, randomFloat(global_id.x * 4) * 20.0 - 10.0, 1.0), // speed
    vec4<f32>(0.0, 0.0, 0.0, 0.0), /* accel */

    vec4<f32>(0.0, 0.0, 0.0, 0.0) /* accel */);
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;

    let target_pos = dimensions.target_pos.xyz;

    if (in_vertex_index == 0u) {
        // Draw cursor at target_pos
        out.clip_position = dimensions.proj * dimensions.view * vec4<f32>(target_pos, 1.0);
        out.color = vec4<f32>(1.0, 0.0, 0.0, 1.0);
        return out;
    }

    let particle = particles[in_vertex_index];
    out.clip_position = dimensions.proj * dimensions.view * vec4<f32>(particle.pos.xyz, 1.0);
    out.color = vec4<f32>(0.0, 0.0, 0.0, 1.0);

    // Point size is not supported in WebGPU directly in the vertex shader (needs point-list topology)
    // But we can just draw points.

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}