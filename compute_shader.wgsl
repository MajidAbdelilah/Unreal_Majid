struct Dimensions {
    width: u32,
    height: u32,
    stride: u32,
    num_of_particles: u32,
    frame_time: f32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
    proj: mat4x4<f32>,
    view: mat4x4<f32>,
}

struct particle {
    pos: vec4<f32>,
    speed: vec4<f32>,
    accel: vec4<f32>,
}

@group(0) @binding(0)
var<storage, read_write> output_buffer: array<u32>;

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

    let x: u32 = u32(particles[global_id.x].pos.x);
    let y: u32 = u32(particles[global_id.x].pos.y);
    if (x > dimensions.width || x < 0 || y > dimensions.height || y < 0) {
        return;
    }

    let index: u32 = y * dimensions.stride + x;

    // Dummy usage to prevent optimization
    // if (index == 0u) {
    //     particles[0].pos = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    // }

    // Alpha = 255
    var color: u32 = 0xFF000000u;

    var left: bool = x < dimensions.width / 2u;
    var right: bool = !left;
    var up: bool = y < dimensions.height / 2u;
    var down: bool = !up;

    // red mask
    var left_up_mask: u32 = u32(left & up) * 0xffffffffu;
    var left_down_mask: u32 = u32(left & down) * 0xffffffffu;

    var right_up_mask: u32 = u32(right & up) * 0xffffffffu;
    var right_down_mask: u32 = u32(right & down) * 0xffffffffu;

    color |= 0x000000FFu & left_up_mask;
    // Red (0xAABBGGRR) -> 0xFF0000FF
    color |= 0x0000FF00u & left_down_mask;
    // Green
    color |= 0x00FF0000u & right_up_mask;
    // Blue
    color |= 0x0000FFFFu & right_down_mask;
    // Yellow (Red + Green) -> 0xFF00FFFF

    output_buffer[index] = color;
}

@compute @workgroup_size(256)
fn init(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x > dimensions.num_of_particles) {
        return;
    }

    particles[global_id.x] = particle(vec4<f32>(randomFloat(global_id.x) * 50.0 + 50.0, randomFloat(global_id.x * 2) * 50.0 + 50.0, randomFloat(global_id.x * 3) * 50.0 - 500.0, 1.0), // pos
    vec4<f32>(0.0), // speed
    vec4<f32>(randomFloat(global_id.x) * 20.0 - 10.0, randomFloat(global_id.x * 2) * 20.0 - 10.0, 0.0, 0.0) /* accel */);
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
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