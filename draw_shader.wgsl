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

fn clamp_color(value: f32, min_val: f32, max_val: f32) -> f32 {
    return min(value * f32(value > min_val), max_val);
}

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


const ONE_OVER_15 = 1.0 / 15.0;

@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;

    let particle = particles[in_vertex_index];
    if(particle.accel.w <= 0.0f)
    {        // dont drwaw dead particles
        out.clip_position = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        out.color = vec4<f32>(1.0, 1.0, 1.0, 0.0);
        return out;
    }
    out.clip_position = dimensions.proj_view * vec4<f32>(particle.pos.xyz, 1.0);
    let particle_pos_len =  (length((dimensions.target_pos.xyz) - (particle.pos.xyz))) * ONE_OVER_15;
    let intensity_r = 1.0 - ((particle_pos_len * (0.9)));
    let intensity_g = 1.0 - ((particle_pos_len * (0.2)));
    let intensity_b = 1.0 - ((particle_pos_len * (0.01)));

    out.color = vec4<f32>(intensity_r, intensity_g, intensity_b, 1.0);

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
