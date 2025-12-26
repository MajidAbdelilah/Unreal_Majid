use std::sync::Arc;
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
struct Dimensions {
    width: u32,
    height: u32,
    stride: u32,
    _pad: u32,
}

pub struct Ren {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    is_surface_configured: bool,
    pub window: Arc<Window>,
    // shader: wgpu::ShaderModel,
    compute_pipeline: wgpu::ComputePipeline,
    bind_group: Option<wgpu::BindGroup>,
    output_buffer: Option<wgpu::Buffer>,
    dimensions_buffer: wgpu::Buffer,
}

impl Ren {
    pub async fn new(window: Arc<Window>) -> anyhow::Result<Self> {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            #[cfg(not(target_arch = "wasm32"))]
            backends: wgpu::Backends::PRIMARY,
            #[cfg(target_arch = "wasm32")]
            backends: wgpu::Backends::GL,
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
                required_features: wgpu::Features::default(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                required_limits: if cfg!(target_arch = "wasm32") {
                    wgpu::Limits::downlevel_webgl2_defaults()
                } else {
                    wgpu::Limits::defaults()
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

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: None,
            module: &shader,
            entry_point: None,
            compilation_options: Default::default(),
            cache: Default::default(),
        });

        let dimensions_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Dimensions Buffer"),
            size: std::mem::size_of::<Dimensions>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
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
            dimensions_buffer,
        })
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
            self.is_surface_configured = true;

            let unpadded_bytes_per_row = width * 4;
            let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
            let padded_bytes_per_row_padding = (align - unpadded_bytes_per_row % align) % align;
            let padded_bytes_per_row = unpadded_bytes_per_row + padded_bytes_per_row_padding;

            let size = (padded_bytes_per_row * height) as u64;
            let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Output Buffer"),
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            self.output_buffer = Some(output_buffer);

            let dimensions = Dimensions {
                width,
                height,
                stride: padded_bytes_per_row / 4,
                _pad: 0,
            };
            self.queue.write_buffer(
                &self.dimensions_buffer,
                0,
                bytemuck::cast_slice(&[dimensions]),
            );
        }
    }

    pub fn handle_key(&self, event_loop: &ActiveEventLoop, code: KeyCode, is_pressed: bool) {
        match (code, is_pressed) {
            (KeyCode::Escape, true) => event_loop.exit(),
            _ => {}
        }
    }

    pub fn update(&mut self) {}

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        self.window.request_redraw();

        if !self.is_surface_configured || self.output_buffer.is_none() {
            return Ok(());
        }

        let output = self.surface.get_current_texture().unwrap();

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        let bind_group_layout = self.compute_pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compute Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.output_buffer.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.dimensions_buffer.as_entire_binding(),
                },
            ],
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.compute_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            let width = self.config.width;
            let height = self.config.height;
            let x_groups = (width + 15) / 16;
            let y_groups = (height + 15) / 16;

            compute_pass.dispatch_workgroups(x_groups, y_groups, 1);
        }

        let unpadded_bytes_per_row = self.config.width * 4;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_bytes_per_row_padding = (align - unpadded_bytes_per_row % align) % align;
        let padded_bytes_per_row = unpadded_bytes_per_row + padded_bytes_per_row_padding;

        encoder.copy_buffer_to_texture(
            wgpu::TexelCopyBufferInfo {
                buffer: self.output_buffer.as_ref().unwrap(),
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(self.config.height),
                },
            },
            wgpu::TexelCopyTextureInfo {
                texture: &output.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: self.config.width,
                height: self.config.height,
                depth_or_array_layers: 1,
            },
        );

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        return Ok(());
    }
}
