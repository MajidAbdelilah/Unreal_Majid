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
var<storage, read> particles: array<particle>;

@group(0) @binding(2)
var<uniform> dimensions: Dimensions;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;

    let particle = particles[in_vertex_index];
    out.clip_position = dimensions.proj_view * vec4<f32>(particle.pos.xyz, 1.0);
    out.color = vec4<f32>(0.0, 0.0, 0.0, 1.0);

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
