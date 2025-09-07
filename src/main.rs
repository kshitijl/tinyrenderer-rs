mod audio;
mod game;
mod image;
mod mesh;
mod render;

use clap::Parser;
use pixels::{Pixels, SurfaceTexture};
use std::sync::Arc;
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, DeviceId, ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

struct App {
    window: Option<Arc<Window>>,
    pixels: Option<Pixels<'static>>,
    world: game::World,
    started: Instant,
    last_frame: Instant,
    total_frames: u64,
}

impl App {
    fn new(world: game::World) -> Self {
        let started = Instant::now();
        let last_frame = started;
        Self {
            window: None,
            pixels: None,
            world,
            started,
            last_frame,
            total_frames: 0,
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes().with_title("tiny"))
                .unwrap(),
        );

        window
            .set_cursor_grab(winit::window::CursorGrabMode::Locked)
            .unwrap();
        window.set_cursor_visible(false);

        self.window = Some(window.clone());
        let pixels = {
            let window_size = window.inner_size();
            let surface_texture =
                SurfaceTexture::new(window_size.width, window_size.height, window.clone());
            match Pixels::new(
                self.world.width() as u32,
                self.world.width() as u32,
                surface_texture,
            ) {
                Ok(mut pixels) => {
                    pixels.set_scaling_mode(pixels::ScalingMode::Fill);
                    window.request_redraw();
                    Some(pixels)
                }
                Err(err) => {
                    log::error!("pixels::new {}", err);
                    event_loop.exit();
                    None
                }
            }
        };

        self.pixels = pixels
    }

    fn device_event(&mut self, _: &ActiveEventLoop, _: DeviceId, event: DeviceEvent) {
        if let DeviceEvent::MouseMotion { delta } = event {
            let (x, y) = delta;
            self.world.camera_mouse(x, y);
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::KeyboardInput {
                device_id: _,
                event,
                is_synthetic: _,
            } => {
                if event.state == ElementState::Pressed {
                    if event.physical_key == PhysicalKey::Code(KeyCode::Escape) {
                        log::info!("bye");
                        event_loop.exit();
                    } else if let PhysicalKey::Code(key) = event.physical_key
                        && !self.world.keys.contains(&key)
                    {
                        self.world.keys.insert(key);
                        self.world.first_pressed_this_frame.insert(key);
                    }
                } else if event.state == ElementState::Released
                    && let PhysicalKey::Code(key) = event.physical_key
                {
                    self.world.keys.remove(&key);
                }
            }
            WindowEvent::CloseRequested => {
                println!("The close button was pressed; stopping");
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                if let Err(err) = self
                    .pixels
                    .as_mut()
                    .unwrap()
                    .resize_surface(size.width, size.height)
                {
                    log::error!("pixels.resize_surface {}", err);
                    event_loop.exit();
                }
            }
            WindowEvent::RedrawRequested => {
                // Redraw the application.
                //
                // It's preferable for applications that do not render continuously to render in
                // this event rather than in AboutToWait, since rendering in here allows
                // the program to gracefully handle redraws requested by the OS.

                // Draw.

                // Queue a RedrawRequested event.
                //
                // You only need to call this if you've determined that you need to redraw in
                // applications which do not always need to. Applications that redraw continuously
                // can render here instead.

                let since_last_frame = self.last_frame.elapsed();
                let since_start = self.started.elapsed();

                let action = self.world.update(since_last_frame, since_start);

                match action {
                    game::ResolutionChangeAction::DoNothing => {
                        // do nothing
                    }
                    game::ResolutionChangeAction::ChangeTo { x, y } => {
                        self.pixels.as_mut().unwrap().resize_buffer(x, y).unwrap()
                    }
                }

                // Draw the current frame

                let average_fps =
                    self.total_frames as f64 / (self.last_frame - self.started).as_secs_f64();
                let this_frame_fps = 1.0f64 / (since_last_frame.as_secs_f64());
                self.total_frames += 1;

                self.last_frame = Instant::now();
                let rendering_result = self.world.draw(self.pixels.as_mut().unwrap().frame_mut());
                if self.total_frames % 1000 == 0 {
                    log::info!(
                        "{:?} average fps {}, this frame {}",
                        rendering_result,
                        average_fps,
                        this_frame_fps
                    );
                }
                if let Err(err) = self.pixels.as_ref().unwrap().render() {
                    log::error!("pixels.render {}", err);
                    event_loop.exit();
                } else {
                    // Queue a redraw for the next frame
                    self.window.as_ref().unwrap().request_redraw();
                }
            }
            _ => (),
        }
    }
}

#[derive(Parser)]
struct Args {
    #[arg(short, long, default_value = "assets/head.obj")]
    exhibit_models: Vec<String>,

    #[arg(short, long, default_value = "assets/man.obj")]
    guard_model: String,

    #[arg(short, long)]
    wall_model_debug: Option<String>,

    /// Output image size in pixels. We only do square images for now.
    #[arg(short, long, default_value_t = 320)]
    canvas_size: u16,
}

fn main() -> std::io::Result<()> {
    env_logger::init();

    let args = Args::parse();
    let audio_system = audio::AudioSystem::new();
    let world = game::World::new(&args, audio_system);

    let event_loop = EventLoop::new().unwrap();

    // ControlFlow::Poll continuously runs the event loop, even if the OS hasn't
    // dispatched any events. This is ideal for games and similar applications.
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::new(world);
    event_loop.run_app(&mut app).unwrap();

    Ok(())
}
