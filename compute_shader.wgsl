struct Dimensions {
    width: u32,
    height: u32,
    stride: u32,
}

@group(0) @binding(0)
var<storage, read_write> output_buffer: array<u32>;

@group(0) @binding(1)
var<uniform> dimensions: Dimensions;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let coords = global_id.xy;

    if (coords.x >= dimensions.width || coords.y >= dimensions.height) {
        return;
    }

    let index = coords.y * dimensions.stride + coords.x;

    var color: u32 = 0xFF000000u; // Alpha = 255

    if (coords.x < dimensions.width / 2u) {
        if (coords.y < dimensions.height / 2u) {
            color |= 0x000000FFu; // Red (0xAABBGGRR) -> 0xFF0000FF
        } else {
            color |= 0x0000FF00u; // Green
        }
    } else {
        if (coords.y < dimensions.height / 2u) {
            color |= 0x00FF0000u; // Blue
        } else {
            color |= 0x0000FFFFu; // Yellow (Red + Green) -> 0xFF00FFFF
        }
    }

    output_buffer[index] = color;
}
