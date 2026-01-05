use std::sync::Arc;
use std::time::Instant;

#[cfg(target_arch = "wasm32")]
use winit::event_loop::{self, EventLoop};
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, ElementState, KeyEvent, MouseButton, WindowEvent},
    keyboard::{KeyCode, PhysicalKey},
    window::Window,
};

use crate::core::renderer::Ren;
pub struct Win {
    #[cfg(target_arch = "wasm32")]
    proxy: Option<winit::event_loop::EventLoopProxy<Ren>>,
    ren: Option<Ren>,
    last_fps_update: Option<Instant>,
    frame_count: u32,
    last_frame_time: Option<Instant>,
    frame_time: f32,
    pointer_pos: [u32; 2],
    mouse_right_pressed: bool,
}

impl Win {
    pub fn new(#[cfg(target_arch = "wasm32")] event_loop: &EventLoop<Ren>) -> Self {
        #[cfg(target_arch = "wasm32")]
        let proxy = Some(event_loop.create_proxy());
        Self {
            ren: None,
            #[cfg(target_arch = "wasm32")]
            proxy,
            last_fps_update: None,
            frame_count: 0,
            last_frame_time: None,
            frame_time: 0.0,
            pointer_pos: [0, 0],
            mouse_right_pressed: false,
        }
    }
}

impl ApplicationHandler<Ren> for Win {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        #[allow(unused_mut)]
        let mut window_attributes = Window::default_attributes();
        // .with_fullscreen(Some(winit::window::Fullscreen::Borderless(None)));

        #[cfg(target_arch = "wasm32")]
        {
            use wasm_bindgen::JsCast;
            use winit::platform::web::WindowAttributesExtWebSys;

            const CANVAS_ID: &str = "canvas";

            let window = wgpu::web_sys::window().unwrap();
            let document = window.document().unwrap();
            let canvas = document.get_element_by_id(CANVAS_ID).unwrap();
            let html_canvas_element = canvas.unchecked_into();
            window_attributes = window_attributes.with_canvas(Some(html_canvas_element));
        }

        let window = Arc::new(event_loop.create_window(window_attributes).unwrap());

        #[cfg(not(target_arch = "wasm32"))]
        {
            // use pollster to await
            self.ren = Some(pollster::block_on(Ren::new(window)).unwrap());
        }

        #[cfg(target_arch = "wasm32")]
        {
            if let Some(proxy) = self.proxy.take() {
                wasm_bindgen_futures::spawn_local(async move {
                    assert!(
                        proxy
                            .send_event(Ren::new(window).await.expect("unable to create canvas"))
                            .is_ok()
                    )
                });
            }
        }
    }

    fn user_event(&mut self, event_loop: &winit::event_loop::ActiveEventLoop, mut event: Ren) {
        #[cfg(target_arch = "wasm32")]
        {
            event.window.request_redraw();
            event.resize(
                event.window.inner_size().width,
                event.window.inner_size().height,
            );
        }
        self.ren = Some(event);
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        let ren = match &mut self.ren {
            Some(canvas) => canvas,
            None => return,
        };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => ren.resize(size.width, size.height),
            WindowEvent::RedrawRequested => {
                self.frame_count += 1;
                if !self.last_frame_time.is_none() {
                    let now = Instant::now();
                    let elapsed = now.duration_since(self.last_frame_time.unwrap());
                    self.frame_time = elapsed.as_secs_f32();
                }
                self.last_frame_time = Some(Instant::now());

                if self.last_fps_update.is_none() {
                    self.last_fps_update = Some(Instant::now());
                }

                if let Some(last) = self.last_fps_update {
                    let now = Instant::now();
                    let elapsed = now.duration_since(last);
                    if elapsed.as_secs_f32() >= 1.0 {
                        let fps = self.frame_count as f32 / elapsed.as_secs_f32();
                        let title = format!("Unreal Majid - FPS: {:.2}", fps);
                        ren.window.set_title(&title);

                        self.frame_count = 0;
                        self.last_fps_update = Some(now);
                    }
                }

                ren.update();
                match ren.render(self.frame_time, self.pointer_pos) {
                    Ok(_) => {}
                    Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                        let size = ren.window.inner_size();
                        ren.resize(size.width, size.height);
                    }
                    Err(e) => {
                        log::error!("Unable to render {}", e);
                    }
                }
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state: key_state,
                        ..
                    },
                ..
            } => ren.handle_key(event_loop, code, key_state.is_pressed()),
            WindowEvent::CursorMoved {
                device_id,
                position,
            } => {
                self.pointer_pos[0] = position.x as u32;
                self.pointer_pos[1] = position.y as u32;
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if button == MouseButton::Left {
                    self.mouse_right_pressed = state == ElementState::Pressed;
                }
            }
            _ => {}
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &winit::event_loop::ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: winit::event::DeviceEvent,
    ) {
        if let Some(ren) = &mut self.ren {
            if let DeviceEvent::MouseMotion { delta } = event {
                if self.mouse_right_pressed {
                    ren.handle_mouse_motion(delta.0, delta.1);
                }
            }
        }
    }

    fn about_to_wait(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        // if let Some(ren) = &self.ren {
        //     let target_fps = 10000.0;
        //     let frame_duration = std::time::Duration::from_secs_f32(1.0 / target_fps);

        //     if let Some(last_frame) = self.last_frame_time {
        //         let now = Instant::now();
        //         if now.duration_since(last_frame) >= frame_duration {
        //             ren.window.request_redraw();
        //         } else {
        //             event_loop.set_control_flow(winit::event_loop::ControlFlow::WaitUntil(
        //                 last_frame + frame_duration,
        //             ));
        //         }
        //     } else {
        //         ren.window.request_redraw();
        //     }
        // }
    }
}
