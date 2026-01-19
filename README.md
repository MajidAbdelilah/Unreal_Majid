# Unreal Majid - Particle System

A high-performance GPU particle simulation written in Rust using [wgpu](https://wgpu.rs/). This project demonstrates the power of Compute Shaders to handle physics calculations for millions of particles in real-time, rendering them efficiently on both native desktop platforms and the web via WebAssembly.

## Live Demo

[Run in Browser](https://majidabdelilah.github.io/Unreal_Majid/)

**Note**: You need a desktop and a browser that support **WebGPU**. Mobile devices are not supported.

## Features

- **GPU Compute Shaders**: Physics calculations (position, velocity, acceleration) are offloaded to the GPU for massive parallelism.
- **Cross-Platform**: Runs natively on Windows/Linux/macOS and in modern web browsers supporting WebGPU.
- **Interactive Camera**: Fly-cam style movement with mouse-look controls.
- **Dynamic Simulation**:
  - **Gravity Control**: Toggle gravity on/off.
  - **Mouse Attraction**: Make gravity pull particles towards the mouse cursor.
  - **Shape Resets**: instantly arrange particles into Sphere or Cube formations.

## Controls

| Key | Action |
| --- | --- |
| **W** | Move Camera Forward |
| **S** | Move Camera Backward |
| **A** | Move Camera Left |
| **D** | Move Camera Right |
| **Space** / **Q** | Move Camera Up |
| **Left Shift** / **E** | Move Camera Down |
| **R** | Reset particles to **Sphere** shape |
| **T** | Reset particles to **Cube** shape |
| **G** | Toggle Gravity |
| **F** | Toggle Gravity Follows Mouse |
| **Esc** | Exit Application |

**Mouse**: Click and hold **Left Mouse Button** + Drag to rotate the camera.

## Building and Running

### Prerequisites

- [Rust Toolchain](https://www.rust-lang.org/tools/install) (cargo, rustc)
- [wasm-pack](https://rustwasm.github.io/wasm-pack/installer.html) (for Web builds)

### Native (Desktop)

To run the simulation natively on your machine:

1.  **Run with Cargo**:
    ```bash
    cargo run --release
    ```

    *Note: The `--release` flag is highly recommended for performance, especially with many particles.*

2.  **Using Makefile**:
    ```bash
    make
    ./target/release/particle-system
    ```

### Web (WebAssembly)

To build and view the project in a browser:

1.  **Build**:
    ```bash
    wasm-pack build --target web --out-name Unreal_Majid
    ```

2.  **Serve**:
    Due to browser security restrictions (CORS), you must serve the files via a local web server.

    ```bash
    # Using Python 3
    python3 -m http.server
    ```

3.  **View**:
    Open your browser to `http://localhost:8000`. Ensure your browser supports WebGPU.

## Project Structure

- **`src/main.rs`**: Entry point for native desktop execution.
- **`src/lib.rs`**: Entry point for WASM/Web execution.
- **`src/core/renderer.rs`**: Handles WGPU context, pipeline setup, and the main render loop.
- **`src/core/window.rs`**: Manages the application window and input events using `winit`.
- **`compute_shader.wgsl`**: WGSL shader responsible for particle physics updates.
- **`draw_shader.wgsl`**: WGSL shader responsible for drawing the particles.