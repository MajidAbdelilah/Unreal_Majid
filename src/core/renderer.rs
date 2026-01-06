use std::{default, process::exit, sync::Arc};
use wgpu::util::DeviceExt;

use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::Window,
};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Mat4f {
    data: [[f32; 4]; 4],
}

impl std::ops::Mul for Mat4f {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        let mut data = [[0.0; 4]; 4];
        for c in 0..4 {
            for r in 0..4 {
                data[c][r] = self.data[0][r] * rhs.data[c][0]
                    + self.data[1][r] * rhs.data[c][1]
                    + self.data[2][r] * rhs.data[c][2]
                    + self.data[3][r] * rhs.data[c][3];
            }
        }
        Mat4f { data }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Dimensions {
    width: u32,
    height: u32,
    stride: u32,
    num_of_particles: u32,
    frame_time: f32,
    _pad: [u32; 3],
    target_pos: [f32; 4],
    proj_view: Mat4f,
}

#[derive(Default)]
pub struct InputState {
    pub forward: bool,
    pub backward: bool,
    pub left: bool,
    pub right: bool,
    pub up: bool,
    pub down: bool,
}

pub struct Camera {
    pub position: [f32; 3],
    pub yaw: f32,
    pub pitch: f32,
}

impl Camera {
    pub fn new() -> Self {
        Self {
            position: [0.0, 0.0, -200.0],
            yaw: 90.0f32.to_radians(), // Pointing along +Z? Old code: -200 to 0 -> +Z direction.
            // If yaw=0 is +X, yaw=90 is +Z?
            // cos(-90)=0, sin(-90)=-1 (if Z=sin).
            // Let's stick to standard math.
            pitch: 0.0,
        }
    }

    pub fn get_view_matrix(&self) -> Mat4f {
        let (sin_y, cos_y) = self.yaw.sin_cos();
        let (sin_p, cos_p) = self.pitch.sin_cos();

        // Forward vector (Standard OpenGL: Camera looks down -Z by default, so 'forward' is -Z in view space)
        // But in World Space, we define where it looks.
        // Y is Up.
        let forward = [cos_p * cos_y, sin_p, cos_p * sin_y];
        let forward = vec3_normalize(forward);

        // Target = pos + forward
        let target = vec3_add(&self.position, &forward);

        look_at(&self.position, &target, &[0.0, 1.0, 0.0])
    }

    pub fn get_forward(&self) -> [f32; 3] {
        let (sin_y, cos_y) = self.yaw.sin_cos();
        let (sin_p, cos_p) = self.pitch.sin_cos();
        let f = [cos_p * cos_y, sin_p, cos_p * sin_y];
        vec3_normalize(f)
    }

    pub fn get_right(&self) -> [f32; 3] {
        let f = self.get_forward();
        let up = [0.0, 1.0, 0.0];
        vec3_normalize(cross(&f, &up)) // Cross logic: Right Hand?
    }
}

fn vec3_add(a: &[f32; 3], b: &[f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn vec3_sub(a: &[f32; 3], b: &[f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn vec3_scale(a: &[f32; 3], s: f32) -> [f32; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Particle {
    pos: [f32; 4],
    speed: [f32; 4],
    accel: [f32; 4],
}

fn dot(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

// Compute the cross product of two [f32; 3] vectors.
fn cross(a: &[f32; 3], b: &[f32; 3]) -> [f32; 3] {
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ];
}

pub fn vec3_normalize(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len > 0.0 {
        [v[0] / len, v[1] / len, v[2] / len]
    } else {
        v
    }
}

fn look_at(&position: &[f32; 3], &lookat: &[f32; 3], &up: &[f32; 3]) -> Mat4f {
    let mut look_at = lookat.clone();
    look_at[0] -= position[0];
    look_at[1] -= position[1];
    look_at[2] -= position[2];

    let f: &[f32; 3] = &vec3_normalize(look_at); // Camera's direction vector
    let s: &[f32; 3] = &vec3_normalize(cross(f, &up)); // Camera's right vector
    let u: &[f32; 3] = &cross(s, f); // Camera's corrected up vector

    let mut result: Mat4f = Mat4f {
        data: [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    };
    result.data[0][0] = s[0];
    result.data[0][1] = u[0];
    result.data[0][2] = -f[0];
    result.data[0][3] = 0.0;

    result.data[1][0] = s[1];
    result.data[1][1] = u[1];
    result.data[1][2] = -f[1];
    result.data[1][3] = 0.0;

    result.data[2][0] = s[2];
    result.data[2][1] = u[2];
    result.data[2][2] = -f[2];
    result.data[2][3] = 0.0;

    result.data[3][0] = -dot(s, &position);
    result.data[3][1] = -dot(u, &position);
    result.data[3][2] = dot(f, &position);
    result.data[3][3] = 1.0;

    return result;
}

#[allow(dead_code)]
fn identity() -> Mat4f {
    Mat4f {
        data: [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    }
}

// Replace the old perspective(...) with a simpler, OpenGL-friendly one.
// Now takes fov_y (radians) and aspect ratio, returns column-major mat4 (data[column][row]).
fn perspective(fov_y: f32, aspect: f32, near: f32, far: f32) -> Mat4f {
    // f = 1 / tan(fov_y/2)
    let f = 1.0 / (fov_y * 0.5).tan();

    let mut m = Mat4f {
        data: [[0.0; 4]; 4],
    };

    // Column-major layout: data[column][row]
    // Column 0
    m.data[0][0] = f / aspect;
    m.data[0][1] = 0.0;
    m.data[0][2] = 0.0;
    m.data[0][3] = 0.0;

    // Column 1
    m.data[1][0] = 0.0;
    m.data[1][1] = f;
    m.data[1][2] = 0.0;
    m.data[1][3] = 0.0;

    // Column 2
    // WebGPU uses Z in [0, 1]
    // z_ndc = far / (near - far) * z_eye + (far * near) / (near - far) * w_eye
    m.data[2][0] = 0.0;
    m.data[2][1] = 0.0;
    m.data[2][2] = far / (near - far);
    m.data[2][3] = -1.0;

    // Column 3
    m.data[3][0] = 0.0;
    m.data[3][1] = 0.0;
    m.data[3][2] = (far * near) / (near - far);
    m.data[3][3] = 0.0;

    m
}

pub struct Ren {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    is_surface_configured: bool,
    pub window: Arc<Window>,
    compute_pipeline: wgpu::ComputePipeline,
    rendering_pipeline: wgpu::RenderPipeline,
    bind_group: Option<wgpu::BindGroup>,
    output_buffer: Option<wgpu::Buffer>,
    particles: wgpu::Buffer,
    num_of_particles: u32,
    dimensions_buffer: wgpu::Buffer,
    frame_time: f32,
    size: (u32, u32),
    padded_bytes_per_row: u32,
    pointer_pos: [u32; 2],
    projection: Mat4f,
    pub camera: Camera,
    pub input_state: InputState,
}

impl Ren {
    pub async fn new(window: Arc<Window>) -> anyhow::Result<Self> {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            #[cfg(not(target_arch = "wasm32"))]
            backends: wgpu::Backends::PRIMARY,
            #[cfg(target_arch = "wasm32")]
            backends: wgpu::Backends::BROWSER_WEBGPU,
            ..Default::default()
        });

        let surface = instance.create_surface(window.clone()).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        println!(
            "device_info: {:?}, texture features: {:?}, capabilities: {:?}",
            adapter.get_info(),
            adapter.get_texture_format_features(wgpu::TextureFormat::Rgba8Unorm),
            adapter.get_downlevel_capabilities(),
        );

        let formats_to_check = [
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureFormat::Bgra8Unorm,
            wgpu::TextureFormat::Rgba16Float,
            wgpu::TextureFormat::Rgba32Float,
        ];

        println!("Checking texture format capabilities:");
        for format in formats_to_check {
            let features = adapter.get_texture_format_features(format);
            println!(
                "Format {:?}: allowed_usages={:?}, flags={:?}",
                format, features.allowed_usages, features.flags
            );
            if features
                .allowed_usages
                .contains(wgpu::TextureUsages::STORAGE_BINDING)
            {
                println!("  -> Supports STORAGE_BINDING");
            }
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            let adapters = instance.enumerate_adapters(wgpu::Backends::all());
            println!("for_each adapter: ");
            adapters.iter().for_each(|adapter| {
                println!("device_info: {:?}", adapter.get_info());
            });
        }

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features {
                    features_wgpu: { wgpu::FeaturesWGPU::VERTEX_WRITABLE_STORAGE },
                    ..wgpu::Features::default()
                },
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                required_limits: if cfg!(target_arch = "wasm32") {
                    wgpu::Limits::downlevel_webgl2_defaults()
                } else {
                    wgpu::Limits {
                        max_storage_buffer_binding_size: 1024 * 1024 * 1024 - 4, // 1GB
                        ..wgpu::Limits::defaults()
                    }
                },
                memory_hints: Default::default(),
                trace: wgpu::Trace::Off,
            })
            .await
            .unwrap();

        let surface_caps = surface.get_capabilities(&adapter);

        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| **f == wgpu::TextureFormat::Rgba8Unorm)
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_DST,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::AutoNoVsync,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        let shader = device.create_shader_module(wgpu::include_wgsl!("../../compute_shader.wgsl"));
        let draw_shader =
            device.create_shader_module(wgpu::include_wgsl!("../../draw_shader.wgsl"));

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("init"),
            compilation_options: Default::default(),
            cache: Default::default(),
        });

        let ren_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("rendering pipeline"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &draw_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::PointList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &draw_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview: None,
            cache: None,
        });

        let dimensions_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Dimensions Buffer"),
            size: std::mem::size_of::<Dimensions>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut _num_of_particles = 1000000u32;

        #[cfg(not(target_arch = "wasm32"))]
        {
            let args: Vec<String> = std::env::args().collect();
            if args.len() != 2 {
                println!(
                    "Error: please give me the number of particles as an argument of the program, as follow"
                );
                println!("./name_of_program 1000000");
                exit(0);
            }
            let num_of_particles_res = args[1].parse::<u32>();
            match num_of_particles_res {
                Ok(num) => {
                    _num_of_particles = num;
                }
                Err(e) => {
                    println!("Error: please enter a correct number");
                    exit(0);
                }
            }
        }

        let particle_size = std::mem::size_of::<Particle>() as u64;
        let particles_buffer_size = _num_of_particles as u64 * particle_size;

        let particles = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particles buffer"),
            size: particles_buffer_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let projection = perspective(
            90.0f32.to_radians(),
            size.width as f32 / size.height as f32,
            0.01,
            10000.0,
        );

        queue.write_buffer(
            &dimensions_buffer,
            0,
            bytemuck::cast_slice(&[Dimensions {
                width: size.width,
                height: size.height,
                stride: 0,
                num_of_particles: _num_of_particles,
                frame_time: 0.0,
                _pad: [0; 3],
                target_pos: [0.0; 4],
                proj_view: projection
                    * look_at(&[0.0, 0.0, -200.0], &[0.0, 0.0, 0.0], &[0.0, 1.0, 0.0]),
            }]),
        );

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder init"),
        });

        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compute Bind Group init"),
            layout: &bind_group_layout,
            entries: &[
                // wgpu::BindGroupEntry {
                //     binding: 0,
                //     resource: particles.as_entire_binding(),
                // },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: particles.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: dimensions_buffer.as_entire_binding(),
                },
            ],
        }));

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass init"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            let size = _num_of_particles;
            let x_groups = (size + 255) / 256;

            compute_pass.dispatch_workgroups(x_groups, 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: Default::default(),
        });

        Ok(Self {
            surface,
            device,
            queue,
            config,
            is_surface_configured: false,
            window,
            compute_pipeline: pipeline,
            bind_group: None,
            output_buffer: None,
            particles,
            num_of_particles: _num_of_particles,
            dimensions_buffer,
            frame_time: 0.0,
            size: (size.width, size.height),
            padded_bytes_per_row: 0,
            rendering_pipeline: ren_pipeline,
            pointer_pos: [0, 0],
            projection,
            camera: Camera::new(),
            input_state: InputState::default(),
        })
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.window.request_redraw();

        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;
            self.size = (width, height);
            self.surface.configure(&self.device, &self.config);
            self.is_surface_configured = true;

            self.projection = perspective(
                90.0f32.to_radians(),
                width as f32 / height as f32,
                0.01,
                10000.0,
            );

            let unpadded_bytes_per_row = width * 4;
            let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
            let padded_bytes_per_row_padding = (align - unpadded_bytes_per_row % align) % align;
            let padded_bytes_per_row = unpadded_bytes_per_row + padded_bytes_per_row_padding;
            self.padded_bytes_per_row = padded_bytes_per_row;
            let size = (padded_bytes_per_row * height) as u64;
            self.output_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Output Buffer"),
                size,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));

            self.queue.write_buffer(
                &self.dimensions_buffer,
                0,
                bytemuck::cast_slice(&[Dimensions {
                    width,
                    height,
                    stride: padded_bytes_per_row / 4,
                    num_of_particles: self.num_of_particles,
                    frame_time: self.frame_time,
                    _pad: [0; 3],
                    target_pos: [0.0; 4],
                    proj_view: self.projection * self.camera.get_view_matrix(),
                }]),
            );
        }
    }

    pub fn handle_key(&mut self, event_loop: &ActiveEventLoop, code: KeyCode, is_pressed: bool) {
        match code {
            KeyCode::KeyW => self.input_state.forward = is_pressed,
            KeyCode::KeyS => self.input_state.backward = is_pressed,
            KeyCode::KeyA => self.input_state.left = is_pressed,
            KeyCode::KeyD => self.input_state.right = is_pressed,
            KeyCode::KeyQ | KeyCode::Space => self.input_state.up = is_pressed,
            KeyCode::KeyE | KeyCode::ShiftLeft => self.input_state.down = is_pressed,
            KeyCode::Escape => {
                if is_pressed {
                    event_loop.exit()
                }
            }
            _ => {}
        }
    }

    pub fn handle_mouse_motion(&mut self, delta_x: f64, delta_y: f64) {
        let sensitivity = 0.002;
        self.camera.yaw += delta_x as f32 * sensitivity;
        self.camera.pitch -= delta_y as f32 * sensitivity; // Invert Y usually

        let limit = std::f32::consts::FRAC_PI_2 - 0.01;
        self.camera.pitch = self.camera.pitch.clamp(-limit, limit);
    }

    pub fn update(&mut self) {
        let speed = 200.0 * self.frame_time;

        let forward = self.camera.get_forward();
        let right = self.camera.get_right();
        let up = [0.0, 1.0, 0.0];

        if self.input_state.forward {
            self.camera.position = vec3_add(&self.camera.position, &vec3_scale(&forward, speed));
        }
        if self.input_state.backward {
            self.camera.position = vec3_sub(&self.camera.position, &vec3_scale(&forward, speed));
        }
        if self.input_state.right {
            self.camera.position = vec3_add(&self.camera.position, &vec3_scale(&right, speed));
        }
        if self.input_state.left {
            self.camera.position = vec3_sub(&self.camera.position, &vec3_scale(&right, speed));
        }
        if self.input_state.up {
            self.camera.position = vec3_add(&self.camera.position, &vec3_scale(&up, speed));
        }
        if self.input_state.down {
            self.camera.position = vec3_sub(&self.camera.position, &vec3_scale(&up, speed));
        }
    }

    pub fn render(
        &mut self,
        frame_time: f32,
        pointer_pos: [u32; 2],
    ) -> Result<(), wgpu::SurfaceError> {
        self.window.request_redraw();
        self.frame_time = frame_time;
        self.pointer_pos = pointer_pos;

        // Calculate Target Pos (Raycasting)
        let width = self.size.0 as f32;
        let height = self.size.1 as f32;
        // Map pointer to NDC (-1 to 1)
        let ndc_x = (pointer_pos[0] as f32 / width) * 2.0 - 1.0;
        let ndc_y = 1.0 - (pointer_pos[1] as f32 / height) * 2.0; // Y Up

        let fov = 90.0f32.to_radians();
        let tan_half_fov = (fov * 0.5).tan();
        let aspect = width / height;

        let scale_y = tan_half_fov;
        let scale_x = tan_half_fov * aspect;

        // View Space Ray (Looking down -Z)
        let dir_view = [ndc_x * scale_x, ndc_y * scale_y, -1.0];

        let f = self.camera.get_forward();
        let r = self.camera.get_right();
        let cam_u = cross(&r, &f); // Camera Up

        // Transform to World Space
        // V_world = x*R + y*U + z*(-F)
        let ray_dir = [
            dir_view[0] * r[0] + dir_view[1] * cam_u[0] + dir_view[2] * -f[0],
            dir_view[0] * r[1] + dir_view[1] * cam_u[1] + dir_view[2] * -f[1],
            dir_view[0] * r[2] + dir_view[1] * cam_u[2] + dir_view[2] * -f[2],
        ];
        let ray_dir = vec3_normalize(ray_dir);

        let p = vec3_add(&self.camera.position, &vec3_scale(&ray_dir, 100.0));
        let target_pos = [p[0], p[1], p[2], 0.0];

        self.queue.write_buffer(
            &self.dimensions_buffer,
            0,
            bytemuck::cast_slice(&[Dimensions {
                width: self.size.0,
                height: self.size.1,
                stride: self.padded_bytes_per_row / 4,
                num_of_particles: self.num_of_particles,
                frame_time: self.frame_time,
                _pad: [0; 3],
                target_pos,
                proj_view: self.projection * self.camera.get_view_matrix(),
            }]),
        );

        if !self.is_surface_configured || self.output_buffer.is_none() {
            return Ok(());
        }

        //compute
        let output = self.surface.get_current_texture().unwrap();

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("compute Encoder"),
            });

        // encoder.clear_buffer(self.output_buffer.as_ref().unwrap(), 0, None);

        let bind_group_layout = self.compute_pipeline.get_bind_group_layout(0);
        self.bind_group = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compute Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.particles.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.dimensions_buffer.as_entire_binding(),
                },
            ],
        }));

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.compute_pipeline);
            compute_pass.set_bind_group(0, &self.bind_group, &[]);

            let size = self.num_of_particles;
            let x_groups = (size + 255) / 256;

            compute_pass.dispatch_workgroups(x_groups, 1, 1);
        }

        // let unpadded_bytes_per_row = self.config.width * 4;
        // let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        // let padded_bytes_per_row_padding = (align - unpadded_bytes_per_row % align) % align;
        // let padded_bytes_per_row = unpadded_bytes_per_row + padded_bytes_per_row_padding;

        // encoder.copy_buffer_to_texture(
        //     wgpu::TexelCopyBufferInfo {
        //         buffer: self.output_buffer.as_ref().unwrap(),
        //         layout: wgpu::TexelCopyBufferLayout {
        //             offset: 0,
        //             bytes_per_row: Some(padded_bytes_per_row),
        //             rows_per_image: Some(self.config.height),
        //         },
        //     },
        //     wgpu::TexelCopyTextureInfo {
        //         texture: &output.texture,
        //         mip_level: 0,
        //         origin: wgpu::Origin3d::ZERO,
        //         aspect: wgpu::TextureAspect::All,
        //     },
        //     wgpu::Extent3d {
        //         width: self.config.width,
        //         height: self.config.height,
        //         depth_or_array_layers: 1,
        //     },
        // );

        self.queue.submit(std::iter::once(encoder.finish()));

        // rendering
        // let output = self.surface.get_current_texture().unwrap();
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        // encoder.clear_buffer(self.output_buffer.as_ref().unwrap(), 0, None);

        let bind_group_layout = self.rendering_pipeline.get_bind_group_layout(0);
        self.bind_group = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("render Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                // wgpu::BindGroupEntry {
                //     binding: 0,
                //     resource: self.output_buffer.as_ref().unwrap().as_entire_binding(),
                // },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.particles.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.dimensions_buffer.as_entire_binding(),
                },
            ],
        }));

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("render Pass"),
                timestamp_writes: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 1.0,
                            g: 1.0,
                            b: 1.0,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
            });
            render_pass.set_pipeline(&self.rendering_pipeline);
            render_pass.set_bind_group(0, &self.bind_group, &[]);

            // let size = self.num_of_particles;
            // let x_groups = (size + 255) / 256;

            render_pass.draw(0..self.num_of_particles, 0..1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));

        output.present();
        return Ok(());
    }
}
