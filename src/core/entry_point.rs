use winit::event_loop::EventLoop;

use crate::core::window::Win;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

pub fn run() -> anyhow::Result<()> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::init();
    }
    #[cfg(target_arch = "wasm32")]
    {
        console_log::init_with_level(log::Level::Info).unwrap();
    }

    let event_loop = EventLoop::with_user_event().build().unwrap();
    let mut win = Win::new(
        #[cfg(target_arch = "wasm32")]
        &event_loop,
    );

    event_loop.run_app(&mut win).unwrap();

    Ok(())
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
pub fn run_web() -> Result<(), wasm_bindgen::JsValue> {
    console_error_panic_hook::set_once();
    // println!("error: start");

    run().unwrap();
    Ok(())
}
