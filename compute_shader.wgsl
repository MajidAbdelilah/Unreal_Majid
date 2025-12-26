struct Dimensions {
    width: u32,
    height: u32,
    stride: u32,
    num_of_particles: u32,
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

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let coords = global_id.xy;

    if (coords.x >= dimensions.width || coords.y >= dimensions.height) {
        return;
    }

    let index = coords.y * dimensions.stride + coords.x;

    // Dummy usage to prevent optimization
    if (index == 0u) {
        particles[0].pos = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }

    // Alpha = 255
    var color: u32 = 0xFF000000u;

    var left: bool = coords.x < dimensions.width / 2u;
    var right: bool = !left;
    var up: bool = coords.y < dimensions.height / 2u;
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

    particles[global_id.x] = particle(vec4<f32>(1.0, 1.0, 1.0, 1.0), // pos
    vec4<f32>(0.0), // speed
    vec4<f32>(1.0, 0.5, 0.0, 0.0) /* accel */);
}