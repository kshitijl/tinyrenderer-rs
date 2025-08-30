mod image;
mod wavefront_obj;

use crate::image::*;

use clap::Parser;
use error_iter::ErrorIter as _;
use glam::{Mat3, Mat4, Vec2, Vec3, Vec3Swizzles, Vec4, Vec4Swizzles, vec3};
use pixels::{Pixels, SurfaceTexture};
use std::collections::HashMap;
use std::f32;
use std::sync::Arc;
use std::time::{Duration, Instant};
use wavefront_obj::Mesh;
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

struct Object {
    mesh: Mesh,
    pos: Vec3,
    angle_y: f32,
    scale: f32,
}

struct World {
    render_settings: RenderSettings,

    image: Image,
    depths: DepthBuffer,
    light_depths: DepthBuffer,
    width: usize,

    camera: Camera,

    light: Object,
    objects: Vec<Object>,

    keys: HashMap<KeyCode, bool>,

    time_since_start: Duration,
    angle_time: Duration,
    should_rotate: bool,
}

struct RenderSettings {
    wireframe: bool,
    no_triangles: bool,
}

struct RenderingUniforms {
    m_viewport: Mat4,
    m_projection: Mat4,
    m_light_to_world: Mat4,
    m_view: Mat4,
}

enum PositionedLight {
    None,
    At(Vec3),
}

impl World {
    /// Create a new `World` instance that can draw a moving box.
    fn new(args: &Args) -> Self {
        let mut objects = Vec::new();
        let mut x = -(args.models.len() as f32);

        for model_filename in args.models.iter() {
            let mut model = wavefront_obj::Mesh::from_file(model_filename.as_str()).unwrap();

            let bb = model.bounding_box();
            log::info!(
                "Parsed model {} with {} vertices, {} faces, {} normals. Bounding box: {:?}. Scale: {}",
                model_filename,
                model.num_vertices(),
                model.num_faces(),
                model.num_normals(),
                bb,
                model.scale()
            );

            model.normalize();

            log::info!(
                "After normalization, bounding box is {:?} and scale is {}",
                model.bounding_box(),
                model.scale()
            );

            let object = Object {
                mesh: model,
                pos: vec3(x, 0., 0.),
                angle_y: 0.,
                scale: 1.,
            };

            objects.push(object);
            x += 2.;
        }

        let image = Image::new(args.canvas_size, args.canvas_size);
        let depths = DepthBuffer::new(args.canvas_size, args.canvas_size);
        let light_depths = DepthBuffer::new(args.canvas_size, args.canvas_size);

        let light = Object {
            mesh: objects[0].mesh.clone(),
            pos: vec3(-4., 1., 0.),
            angle_y: 0.,
            scale: 0.3,
        };

        Self {
            image,
            depths,
            light_depths,
            objects,
            width: args.canvas_size as usize,
            render_settings: RenderSettings {
                wireframe: args.wireframe,
                no_triangles: args.no_triangles,
            },
            camera: Camera {
                pos: vec3(0., 0., 3.),
                dir: vec3(0., 0., -3.).normalize(),
                up: vec3(0., 1., 0.).normalize(),
            },
            keys: HashMap::new(),
            light,
            time_since_start: Duration::from_secs(0),
            angle_time: Duration::from_secs(0),
            should_rotate: true,
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
    fn update(&mut self, since_last_frame: Duration, since_start: Duration) {
        if self.keys.get(&KeyCode::KeyW) == Some(&true) {
            self.move_(Direction::Forward);
        }

        if self.keys.get(&KeyCode::KeyS) == Some(&true) {
            self.move_(Direction::Back);
        }

        if self.keys.get(&KeyCode::KeyA) == Some(&true) {
            self.move_(Direction::Left);
        }

        if self.keys.get(&KeyCode::KeyD) == Some(&true) {
            self.move_(Direction::Right);
        }
        if self.keys.get(&KeyCode::KeyR) == Some(&true) {
            self.should_rotate = true;
        }
        if self.keys.get(&KeyCode::KeyT) == Some(&true) {
            self.should_rotate = false;
        }

        self.time_since_start = since_start;

        if self.should_rotate {
            self.angle_time += since_last_frame;
        }

        for (idx, object) in self.objects.iter_mut().enumerate() {
            let angle = self.angle_time.as_secs_f32() * (idx as f32 + 1.);

            object.angle_y = angle;
        }
        let t = self.time_since_start.as_secs_f32();

        //self.light.pos.x = 15. + 5. * f32::sin(1.0 * t);
        self.light.pos.y = 1.9 * f32::sin(1.0 * t);
        // self.light.pos.z = 7. + f32::cos(1.9 * t);
    }

    fn render_object(
        object: &Object,
        uniforms: &RenderingUniforms,
        light: &PositionedLight,
        image: &mut Option<&mut Image>,
        depths: &mut DepthBuffer,
        render_settings: &RenderSettings,
    ) {
        let RenderingUniforms {
            m_viewport,
            m_projection,
            m_light_to_world,
            m_view,
        } = uniforms;

        let m_scale = Mat4::from_scale(vec3(object.scale, object.scale, object.scale));
        let m_rot = Mat4::from_rotation_y(object.angle_y);
        let m_trans = Mat4::from_translation(object.pos);
        let m_model = m_trans * m_rot * m_scale;

        let m_mvp = m_projection * m_view * m_model;

        let m_normal = m_model.inverse().transpose();

        for face_idx in 0..object.mesh.num_faces() {
            let mut screen_coords: [Vec3; 3] = [Vec3::new(0., 0., 0.); 3];
            let mut world_coords: [Vec3; 3] = [Vec3::new(0.0, 0.0, 0.0); 3];

            for j in 0..3 {
                let model_coordinates = Vec4::from((object.mesh.vertex(face_idx, j), 1.0));

                let clip_coordinates = m_mvp * model_coordinates;

                let normalized_device_coordinates = perspective_divided(clip_coordinates);
                // TODO maybe skip drawing these
                // assert!(normalized_device_coordinates.z >= -1.);
                // assert!(normalized_device_coordinates.z <= 1.);

                screen_coords[j] = (m_viewport * normalized_device_coordinates).xyz();
                world_coords[j] = normalized_device_coordinates.xyz();
            }

            if !render_settings.no_triangles {
                let lighting = match light {
                    PositionedLight::None => None,
                    PositionedLight::At(light_pos) => {
                        // We're going to do lighting by dot-producting the light direction
                        // and normals, so it's really THOSE two that need to be transformed
                        // with respect to each other. It's also very important that we
                        // not normalize or xyz the normals and lighting vectors! Those are
                        // non-linear transforms and break the proof that transforming by
                        // the transpose of the inverse preserves dot products.
                        let mut normals = Vec::new();
                        for i in 0..3 {
                            let normal = object.mesh.normal(face_idx, i);
                            let normal = m_normal * Vec4::from((normal, 0.));
                            normals.push(normal);
                        }
                        let light_dir = (Vec4::from((object.pos, 1.0))
                            - Vec4::from((*light_pos, 1.)))
                        .normalize();
                        let transformed_light_dir = m_light_to_world * light_dir;

                        let lighting = ForLighting {
                            light_dir: transformed_light_dir,
                            na: normals[0],
                            nb: normals[1],
                            nc: normals[2],
                        };

                        Some(lighting)
                    }
                };
                triangle(
                    screen_coords[0],
                    screen_coords[1],
                    screen_coords[2],
                    lighting,
                    image,
                    depths,
                );
            }

            if let Some(image) = image
                && render_settings.wireframe
            {
                for i in 0..3 {
                    linevf32(
                        screen_coords[i % 3].xy(),
                        screen_coords[(i + 1) % 3].xy(),
                        image,
                        RED,
                    );
                }
            }
        }
    }

    fn render(&mut self) {
        let canvas_size = self.width as f32;

        let m_view = Mat4::look_to_rh(self.camera.pos, self.camera.dir, self.camera.up);
        let z_near = 1.;
        let z_far = 10.;
        let m_projection = Mat4::perspective_rh_gl(f32::to_radians(60.), 1.0, z_near, z_far);
        let m_light_to_world = Mat4::IDENTITY;

        let m_viewport = Mat4::from_scale(Vec3::new(canvas_size / 2.0, canvas_size / 2.0, 1.))
            * Mat4::from_translation(Vec3::new(1.0, 1.0, 0.0));

        let uniforms = RenderingUniforms {
            m_viewport,
            m_projection,
            m_view,
            m_light_to_world,
        };

        Self::render_object(
            &self.light,
            &uniforms,
            &PositionedLight::None,
            &mut Some(&mut self.image),
            &mut self.depths,
            &self.render_settings,
        );

        for object in self.objects.iter() {
            Self::render_object(
                object,
                &uniforms,
                &PositionedLight::At(self.light.pos),
                &mut Some(&mut self.image),
                &mut self.depths,
                &self.render_settings,
            );
        }

        // Now again from the light's POV

        let m_view = Mat4::look_at_rh(self.light.pos, self.objects[0].pos, self.camera.up);

        let uniforms = RenderingUniforms {
            m_viewport,
            m_projection,
            m_view,
            m_light_to_world,
        };
        for object in self.objects.iter() {
            Self::render_object(
                object,
                &uniforms,
                &PositionedLight::At(self.light.pos),
                &mut None,
                &mut self.light_depths,
                &self.render_settings,
            );
        }
    }

    fn clear(&mut self) {
        // TODO put this clearing code in Image and DepthBuffer respectively
        let data = self.image.buf_mut();
        let u32_slice = unsafe {
            std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u32, data.len() / 4)
        };

        let pattern = 0xaaaaaaffu32;
        u32_slice.fill(pattern);

        let depth_data = self.depths.buf_mut();
        depth_data.as_mut_slice().fill(f32::MAX);

        let depth_data = self.light_depths.buf_mut();
        depth_data.as_mut_slice().fill(f32::MAX);
    }

    fn draw(&mut self, frame: &mut [u8]) {
        self.clear();
        self.render();

        frame.fill(255);

        assert!(self.image.width() == self.width);
        assert!(self.image.height() == self.width);
        let image_buf = self.image.buf().as_slice();
        for x in 0..self.width {
            for y in 0..self.width {
                let image_idx = 4 * y * self.width + 4 * x;
                let frame_idx = 4 * y * (self.width * 2) + 4 * x;

                frame[frame_idx..frame_idx + 4]
                    .copy_from_slice(&image_buf[image_idx..image_idx + 4]);
            }
        }

        assert!(self.light_depths.width() == self.width);
        assert!(self.light_depths.height() == self.width);
        let depth_buf = self.light_depths.buf();
        let min_depth = self.light_depths.min_depth();
        let max_depth = self.light_depths.max_depth();

        for x in 0..self.width {
            for y in 0..self.width {
                let image_idx = y * self.width + x;
                let frame_idx = 4 * y * (self.width * 2) + 4 * (x + self.width);

                let depth = depth_buf[image_idx];
                let gray = DepthBuffer::depth_to_u8(depth, min_depth, max_depth);

                let color = [gray, gray, gray, 255];
                frame[frame_idx..frame_idx + 4].copy_from_slice(&color);
            }
        }
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

        self.window = Some(window.clone());
        let pixels = {
            let window_size = window.inner_size();
            let surface_texture =
                SurfaceTexture::new(window_size.width, window_size.height, window.clone());
            match Pixels::new(
                2 * self.world.width as u32,
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
                    } else if let PhysicalKey::Code(key) = event.physical_key {
                        self.world.keys.insert(key, true);
                    }
                } else if event.state == ElementState::Released
                    && let PhysicalKey::Code(key) = event.physical_key
                {
                    self.world.keys.insert(key, false);
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

                let since_last_frame = self.last_frame.elapsed();
                let since_start = self.started.elapsed();

                self.world.update(since_last_frame, since_start);

                // Draw the current frame

                let average_fps =
                    self.total_frames as f64 / (self.last_frame - self.started).as_secs_f64();
                let this_frame_fps = 1.0f64 / (since_last_frame.as_secs_f64());
                if self.total_frames % 60 == 0 {
                    log::info!("average fps {}, this frame {}", average_fps, this_frame_fps);
                }
                self.total_frames += 1;

                self.last_frame = Instant::now();
                self.world.draw(self.pixels.as_mut().unwrap().frame_mut());
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
    0.5 * answer
}

struct ForLighting {
    light_dir: Vec4,
    na: Vec4,
    nb: Vec4,
    nc: Vec4,
}

fn triangle(
    a: Vec3,
    b: Vec3,
    c: Vec3,
    lighting: Option<ForLighting>,
    image: &mut Option<&mut Image>,
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

            if x >= 0 && x < depths.width() as i32 && y >= 0 && y < depths.height() as i32 {
                let x = x as usize;
                let y = y as usize;
                if z < depths.get(x, y) {
                    depths.set(x, y, z);
                    let ambient_intensity = 0.3;

                    let color = match lighting {
                        None => coloru8(255, 255, 255),
                        Some(ForLighting {
                            light_dir,
                            na,
                            nb,
                            nc,
                        }) => {
                            let normal = alpha * na + beta * nb + gamma * nc;
                            let dir_intensity = normal.dot(-light_dir).clamp(0., 1.);
                            let dir_intensity = (dir_intensity * 6.).round() / 6.;
                            let dir_intensity = dir_intensity * (1. - ambient_intensity);

                            let total_intensity = ambient_intensity + dir_intensity;
                            let color = vec3(255., 155., 0.) * total_intensity;
                            color.as_u8vec3()
                        }
                    };

                    if let Some(image) = image {
                        image.set(x, y, color)
                    }
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
    /// Model files
    models: Vec<String>,

    /// Output image size in pixels. We only do square images for now.
    #[arg(short, long, default_value_t = 320)]
    canvas_size: u16,

    /// Draw red wireframe lines
    #[arg(short, long)]
    wireframe: bool,

    /// Don't draw triangles
    #[arg(short, long)]
    no_triangles: bool,

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
