/*
clip coordinates: output of vertex shader, after it has transformed with projection matrix, but BEFORE the GPU does the perspective divide and clipping against the view frustum

overall pipeline:
1. model space. coordinates of vertices as stored in mesh
2. world space. after applying the model matrix (placing the object in the scene)
3. view/eye space. after applying the view matrix (putting everything in the camera's coordinate system)
4. clip space (clip coordinates). after applying the projection matrix (perspective or orthographic). at this stage each vertex is 4d (x,y,z,w).
5. Normalized Device Coordinates (NDC). Divide by w. (x/w, y/w, z/w). Now everything is in the cube [-1,1]^3.
6. window/screen coordinates. After the viewport transform, mapping NDC to pixel positions.
7. NOW rasterizaton happens. Each fragment has interpolated attributes, including depth.

OpenGL convention is a right handed system where the camera looks down the negative Z axis. X axis points right, Y axis points up, Z axis points towards the camera (positive Z values are behind the camera).

In this convention, you typically specify the camera looking towards negative Z, and object you want to see should have negative Z coordinates relative to the camera.

Objects with greater negative Z (i.e., smaller Z) are farther away from the camera.

See https://www.songho.ca/opengl/gl_projectionmatrix.html for a derivation of the perspective projection matrix entries.

In NDC, after perspective divide, we want for the near plane that z_ndc = -1, and for the far plane z_ndc = +1. But note that the near and far planes are at z = -z_near and z = -z_far. z_near and z_far are positive numbers. But we are facing the negative z direction, so the actual planes are at negative z coordinates.

Also, for the depth test, OpenGL remaps [-1,1] -> [0,1]: z_depth = z_ndc/2 + 1/2. The near plane is z = 0.0, far plane is z = 1.0.
*/
mod image;
mod tga;
mod wavefront_obj;

use crate::image::*;
use clap::Parser;
use glam::{Mat3, Mat4, Vec2, Vec3, Vec3Swizzles, Vec4, Vec4Swizzles, vec3};
use std::f32;

use error_iter::ErrorIter as _;
use log;
use pixels::{Pixels, SurfaceTexture};
use std::sync::Arc;
use std::time::{Duration, Instant};
use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, DeviceId, ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

fn log_error<E: std::error::Error + 'static>(method_name: &str, err: E) {
    log::error!("{method_name}() failed: {err}");
    for source in err.sources().skip(1) {
        log::error!("  Caused by: {source}");
    }
}

struct Camera {
    pos: Vec3,
    dir: Vec3,
    up: Vec3,
}

enum Direction {
    Forward,
    Back,
    Right,
    Left,
}

struct World {
    image: Image,
    depths: DepthBuffer,
    model: wavefront_obj::Model,
    width: usize,
    wireframe: bool,
    camera: Camera,
}

impl World {
    /// Create a new `World` instance that can draw a moving box.
    fn new(args: &Args) -> Self {
        let model = wavefront_obj::Model::from_file(args.model.as_str()).unwrap();

        log::info!(
            "Parsed {} vertices, {} faces, {} normals",
            model.num_vertices(),
            model.num_faces(),
            model.num_normals()
        );

        let image = Image::new(args.canvas_size, args.canvas_size);
        let depths = DepthBuffer::new(args.canvas_size, args.canvas_size);

        Self {
            image,
            depths,
            model,
            width: args.canvas_size as usize,
            wireframe: args.wireframe,
            camera: Camera {
                pos: vec3(1., 1., 3.),
                dir: vec3(-1., -1., -3.).normalize(),
                up: vec3(0., 1., 0.).normalize(),
            },
        }
    }

    fn camera_mouse(&mut self, dx: f64, dy: f64) {
        let m = Mat3::from_rotation_y((-dx / 10.).to_radians() as f32);
        self.camera.dir = m * self.camera.dir;

        let m = Mat3::from_rotation_x((-dy / 10.).to_radians() as f32);
        self.camera.dir = m * self.camera.dir;
    }

    fn move_(&mut self, dir: Direction) {
        let speed = 0.1;
        let forward = self.camera.dir.with_y(0.);
        let right = forward.cross(self.camera.up);

        match dir {
            Direction::Forward => {
                self.camera.pos += forward * speed;
            }
            Direction::Back => {
                self.camera.pos -= forward * speed;
            }
            Direction::Right => {
                self.camera.pos += right * speed;
            }
            Direction::Left => {
                self.camera.pos -= right * speed;
            }
        }

        log::info!("now at {}", self.camera.pos);
    }
    fn update(&mut self) {
        // Nothing to do here for now; we don't animate or whatever.
    }

    fn render(&mut self) {
        let canvas_size = self.width as f32;

        let angle = 0.;
        let m_rot = Mat4::from_rotation_y(angle);
        let m_trans = Mat4::from_translation(Vec3::new(0., 0.0, 0.));
        let m_model = m_trans * m_rot;

        let light_dir = Vec3::new(-1., 0., -1.).normalize();
        let m_view = Mat4::look_to_rh(self.camera.pos, self.camera.dir, self.camera.up);

        let z_near = 1.;
        let z_far = 10.;
        let m_projection = Mat4::perspective_rh_gl(f32::to_radians(60.), 1.0, z_near, z_far);

        let m_mvp = m_projection * m_view * m_model;
        let m_mvpit = (m_projection * m_view * m_model).inverse().transpose();

        let m_viewport = Mat4::from_scale(Vec3::new(canvas_size / 2.0, canvas_size / 2.0, 1.))
            * Mat4::from_translation(Vec3::new(1.0, 1.0, 0.0));

        for face_idx in 0..self.model.num_faces() {
            let mut screen_coords: [Vec3; 3] = [Vec3::new(0., 0., 0.); 3];
            let mut world_coords: [Vec3; 3] = [Vec3::new(0.0, 0.0, 0.0); 3];

            for j in 0..3 {
                let model_coordinates = Vec4::from((self.model.vertex(face_idx, j), 1.0));

                let world_coordinates = m_model * model_coordinates;

                let eye_coordinates = &m_view * &world_coordinates;

                // assert!(eye_coordinates.z < 0.);
                // assert!(eye_coordinates.w == 1.0);

                let clip_coordinates = &m_projection * &eye_coordinates;

                let clip_coordinates = m_mvp * model_coordinates;

                let normalized_device_coordinates = perspective_divided(clip_coordinates);
                // TODO maybe skip drawing these
                // assert!(normalized_device_coordinates.z >= -1.);
                // assert!(normalized_device_coordinates.z <= 1.);

                screen_coords[j] = (&m_viewport * &normalized_device_coordinates).xyz();
                world_coords[j] = normalized_device_coordinates.xyz();
            }

            let mut normals = Vec::new();
            for i in 0..3 {
                let normal = self.model.normal(face_idx, i);
                let normal = m_mvpit * Vec4::from((normal, 0.));
                let normal = normal.xyz();
                normals.push(normal);
            }

            triangle(
                screen_coords[0],
                screen_coords[1],
                screen_coords[2],
                light_dir,
                normals[0],
                normals[1],
                normals[2],
                // normal,
                // normal,
                // normal,
                &mut self.image,
                &mut self.depths,
            );

            if self.wireframe {
                for i in 0..3 {
                    linevf32(
                        screen_coords[i % 3].xy(),
                        screen_coords[(i + 1) % 3].xy(),
                        &mut self.image,
                        RED,
                    );
                }
            }
        }
    }

    fn clear(&mut self) {
        let data = self.image.buf_mut();
        let u32_slice = unsafe {
            std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u32, data.len() / 4)
        };

        let pattern = 0xaaaaaaffu32;
        u32_slice.fill(pattern);

        let depth_data = self.depths.buf_mut();
        depth_data.as_mut_slice().fill(f32::MAX);
    }

    fn draw(&mut self, frame: &mut [u8]) {
        self.clear();
        self.render();
        frame.copy_from_slice(self.image.buf().as_slice());
    }
}

struct App {
    window: Option<Arc<Window>>,
    pixels: Option<Pixels<'static>>,
    world: World,
    started: Instant,
    last_frame: Instant,
    total_frames: u64,
}

impl App {
    fn new(world: World) -> Self {
        let started = Instant::now();
        let last_frame = started.clone();
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

        self.window = Some(window.clone());
        let pixels = {
            let window_size = window.inner_size();
            let surface_texture =
                SurfaceTexture::new(window_size.width, window_size.height, window.clone());
            match Pixels::new(
                self.world.width as u32,
                self.world.width as u32,
                surface_texture,
            ) {
                Ok(pixels) => {
                    window.request_redraw();
                    Some(pixels)
                }
                Err(err) => {
                    log_error("pixels::new", err);
                    event_loop.exit();
                    None
                }
            }
        };

        self.pixels = pixels
    }

    fn device_event(&mut self, _: &ActiveEventLoop, _: DeviceId, event: DeviceEvent) {
        match event {
            DeviceEvent::MouseMotion { delta } => {
                let (x, y) = delta;
                self.world.camera_mouse(x, y);
            }

            _ => {}
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::PinchGesture { .. } => {
                // log::info!("pinch");
            }
            WindowEvent::MouseWheel { .. } => {
                // log::info!("mousewheel");
            }
            WindowEvent::MouseInput { .. } => {
                // log::info!("mouseinput");
            }
            WindowEvent::CursorMoved {
                device_id: _,
                position,
            } => {
                // let (x,y) = position.partial_cmp(other)
            }
            WindowEvent::KeyboardInput {
                device_id,
                event,
                is_synthetic,
            } => {
                // log::info!("keyboard {:?} {:?} {}", device_id, event, is_synthetic);
                if event.state == ElementState::Pressed {
                    if event.physical_key == PhysicalKey::Code(KeyCode::Escape) {
                        log::info!("bye");
                        event_loop.exit();
                    } else if event.physical_key == PhysicalKey::Code(KeyCode::KeyW) {
                        self.world.move_(Direction::Forward);
                    } else if event.physical_key == PhysicalKey::Code(KeyCode::KeyS) {
                        self.world.move_(Direction::Back);
                    } else if event.physical_key == PhysicalKey::Code(KeyCode::KeyD) {
                        self.world.move_(Direction::Right);
                    } else if event.physical_key == PhysicalKey::Code(KeyCode::KeyA) {
                        self.world.move_(Direction::Left);
                    }
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
                    log_error("pixels.resize_surface", err);
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

                // Update internal state
                self.world.update();

                // Draw the current frame
                self.world.draw(self.pixels.as_mut().unwrap().frame_mut());

                let average_fps =
                    self.total_frames as f64 / (self.last_frame - self.started).as_secs_f64();
                let this_frame_fps = 1.0f64 / (self.last_frame.elapsed().as_secs_f64());
                if self.total_frames % 60 == 0 {
                    log::info!("average fps {}, this frame {}", average_fps, this_frame_fps);
                }
                self.total_frames += 1;
                self.last_frame = Instant::now();
                if let Err(err) = self.pixels.as_ref().unwrap().render() {
                    log_error("pixels.render", err);
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

fn linei32(ax: i32, ay: i32, bx: i32, by: i32, image: &mut Image, color: Color) {
    let steep = (by - ay).abs() > (bx - ax).abs();
    let (ax, bx, ay, by) = if !steep {
        (ax, bx, ay, by)
    } else {
        (ay, by, ax, bx)
    };

    let (ax, bx, ay, by) = if ax <= bx {
        (ax, bx, ay, by)
    } else {
        (bx, ax, by, ay)
    };

    assert!(ax <= bx);
    assert!((ax - bx).abs() >= (ay - by).abs());

    let mut x = ax;
    let mut y = ay;
    let mut ierror = 0; // defined as error * 2 * (bx - ax)
    let dy = if by > ay { 1 } else { -1 };
    while x <= bx {
        let (xx, yy) = if !steep { (x, y) } else { (y, x) };

        // skip points outside the image bounds. we do this discarding here
        // rather than outside the loop so we draw any visible portions of lines
        // whose endpoints might lie outside bounds.
        if xx >= 0 && yy >= 0 && xx < image.width() as i32 && yy < image.height() as i32 {
            image.set(xx as usize, yy as usize, color);
        }

        ierror += (by - ay).abs() * 2;
        let should_incr = (ierror > (bx - ax)) as i32;
        y += dy * should_incr;
        ierror -= 2 * (bx - ax) * should_incr;
        x += 1;
    }
}

fn linef32(ax: f32, ay: f32, bx: f32, by: f32, image: &mut Image, color: Color) {
    linei32(ax as i32, ay as i32, bx as i32, by as i32, image, color)
}

fn linevf32(a: Vec2, b: Vec2, image: &mut Image, color: Color) {
    linef32(a.x, a.y, b.x, b.y, image, color)
}

fn signed_triangle_area(a: Vec2, b: Vec2, c: Vec2) -> f32 {
    let answer = (b.y - a.y) * (b.x + a.x) + (c.y - b.y) * (c.x + b.x) + (a.y - c.y) * (a.x + c.x);
    0.5 * answer as f32
}

fn triangle(
    a: Vec3,
    b: Vec3,
    c: Vec3,
    light_dir: Vec3,
    na: Vec3,
    nb: Vec3,
    nc: Vec3,
    image: &mut Image,
    depths: &mut DepthBuffer,
) {
    let total_area = signed_triangle_area(a.xy(), b.xy(), c.xy());

    let smallest_x = f32::min(a.x, f32::min(b.x, c.x)) as i32;
    let smallest_y = f32::min(a.y, f32::min(b.y, c.y)) as i32;
    let biggest_x = f32::max(a.x, f32::max(b.x, c.x)) as i32;
    let biggest_y = f32::max(a.y, f32::max(b.y, c.y)) as i32;

    for x in smallest_x..=biggest_x {
        for y in smallest_y..=biggest_y {
            let p = Vec2::new(x as f32, y as f32);

            let alpha = signed_triangle_area(p, b.xy(), c.xy()) / total_area;
            if alpha < 0.0 {
                continue;
            }

            let beta = signed_triangle_area(p, c.xy(), a.xy()) / total_area;
            if beta < 0.0 {
                continue;
            }

            let gamma = signed_triangle_area(p, a.xy(), b.xy()) / total_area;
            if gamma < 0.0 {
                continue;
            }

            let z = alpha * a.z + beta * b.z + gamma * c.z;
            let z = z / 2. + 0.5;
            // assert!(z >= 0.);
            // assert!(z <= 1.);

            if x >= 0 && x < image.width() as i32 && y >= 0 && y < image.height() as i32 {
                let x = x as usize;
                let y = y as usize;
                if z < depths.get(x, y) {
                    depths.set(x, y, z);

                    let normal = alpha * na + beta * nb + gamma * nc;
                    let normal = normal.normalize();
                    let intensity = normal.dot(light_dir).clamp(0., 1.);
                    let intensity = (intensity * 6.).round() / 6.;
                    let color = vec3(255., 155., 0.) * intensity;
                    let color = color.as_u8vec3();

                    image.set(x, y, color);
                }
            }
        }
    }
}

fn perspective_divided(v: Vec4) -> Vec4 {
    // Vec4::new(v.x / v.w, v.y / v.w, v.z / v.w, 1.)
    v / v.w
}

#[derive(Parser)]
struct Args {
    /// Model file
    model: String,

    /// Output image size in pixels. We only do square images for now.
    #[arg(short, long, default_value_t = 800)]
    canvas_size: u16,

    /// Draw red wireframe lines
    #[arg(short, long)]
    wireframe: bool,

    #[arg(long)]
    write_depth_buffer: bool,
}

fn main() -> std::io::Result<()> {
    env_logger::init();

    let args = Args::parse();
    let world = World::new(&args);

    let event_loop = EventLoop::new().unwrap();

    // ControlFlow::Poll continuously runs the event loop, even if the OS hasn't
    // dispatched any events. This is ideal for games and similar applications.
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::new(world);
    event_loop.run_app(&mut app).unwrap();

    // let tga = tga::TgaFile::from_image(image);
    // tga.save_to_path(format!("raw-output/frame-{:02}.tga", idx).as_str())?;

    // if args.write_depth_buffer {
    //     let depth_tga = tga::TgaFile::from_image(depths.to_image());
    //     depth_tga.save_to_path(format!("raw-output/depth-{:02}.tga", idx).as_str())?;
    // }

    Ok(())
}
