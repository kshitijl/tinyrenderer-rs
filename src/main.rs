mod image;
mod wavefront_obj;

use crate::image::*;

use clap::Parser;
use error_iter::ErrorIter as _;
use glam::{Mat3, Mat4, Vec2, Vec3, Vec3Swizzles, Vec4, Vec4Swizzles, vec3};
use pixels::{Pixels, SurfaceTexture};
use std::collections::HashSet;
use std::f32;
use std::ops::Add;
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
    angle_x: f32,
    angle_y: f32,
    scale: f32,
}

struct Spotlight {
    pos: Vec3,
    dir: Vec3,
}

struct World {
    render_settings: RenderSettings,
    movement_settings: MovementSettings,

    image: Image,
    depths: DepthBuffer,
    light_depths: DepthBuffer,
    width: usize,

    camera: Camera,

    light: Spotlight,
    objects: Vec<Object>,

    keys: HashSet<KeyCode>,
    first_pressed_this_frame: HashSet<KeyCode>,

    time_since_start: Duration,
    angle_time: Duration,
}

struct RenderSettings {
    wireframe: bool,
    no_triangles: bool,
    draw_lightbulb: bool,
}

struct MovementSettings {
    rotate_objects: bool,
    move_light_around: bool,
}

struct RenderingUniforms {
    m_viewport: Mat4,
    m_projection: Mat4,
    m_view: Mat4,
}

#[derive(Copy, Clone, PartialEq, Debug)]
struct RenderingResult {
    num_objects_bb_culled: u32,
    num_bb_pixels_considered: u32,
    num_pixels_drawn: u32,
    num_triangle_pixels_considered: u32,
    num_triangles_with_onscreen_bb: u32,
    num_in_bounds_triangle_pixels_considered: u32,
    num_depth_buffer_sets: u32,
    num_unclipped_triangles_considered: u32,
}

impl RenderingResult {
    fn new() -> Self {
        Self {
            num_objects_bb_culled: 0,
            num_triangles_with_onscreen_bb: 0,
            num_pixels_drawn: 0,
            num_in_bounds_triangle_pixels_considered: 0,
            num_depth_buffer_sets: 0,
            num_triangle_pixels_considered: 0,
            num_bb_pixels_considered: 0,
            num_unclipped_triangles_considered: 0,
        }
    }
}

impl Add for RenderingResult {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            num_triangles_with_onscreen_bb: self.num_triangles_with_onscreen_bb
                + other.num_triangles_with_onscreen_bb,
            num_pixels_drawn: self.num_pixels_drawn + other.num_pixels_drawn,
            num_in_bounds_triangle_pixels_considered: self.num_in_bounds_triangle_pixels_considered
                + other.num_in_bounds_triangle_pixels_considered,
            num_depth_buffer_sets: self.num_depth_buffer_sets + other.num_depth_buffer_sets,
            num_triangle_pixels_considered: self.num_triangle_pixels_considered
                + other.num_triangle_pixels_considered,
            num_bb_pixels_considered: self.num_bb_pixels_considered
                + other.num_bb_pixels_considered,
            num_unclipped_triangles_considered: self.num_unclipped_triangles_considered
                + other.num_unclipped_triangles_considered,
            num_objects_bb_culled: self.num_objects_bb_culled + other.num_objects_bb_culled,
        }
    }
}

struct BaryCoords(Vec3);

trait Shader {
    type Varying: Copy;

    fn vertex(&self, coord: Vec4, normal: Vec4) -> Self::Varying;
    fn fragment(&self, varyings: &[Self::Varying; 3], b: BaryCoords) -> Color;
}

struct NoopShaderColorsWhite {
    //
}

impl Shader for NoopShaderColorsWhite {
    type Varying = u32;

    fn vertex(&self, _coord: Vec4, _normal: Vec4) -> u32 {
        1
    }

    fn fragment(&self, _varyings: &[u32; 3], _b: BaryCoords) -> Color {
        coloru8(255, 255, 255)
    }
}

impl NoopShaderColorsWhite {
    fn new() -> Self {
        Self {}
    }
}

struct FinalRenderShaders<'buf> {
    light_pos: Vec3,
    light_vp: Mat4,
    light_viewport: Mat4,
    light_pov_depths: &'buf DepthBuffer,
}

#[derive(Clone, Copy)]
struct FinalRenderVarying {
    world_coord: Vec4,
    normal: Vec4,
}

impl<'buf> Shader for FinalRenderShaders<'buf> {
    type Varying = FinalRenderVarying;

    fn vertex(&self, world_coord: Vec4, normal: Vec4) -> FinalRenderVarying {
        FinalRenderVarying {
            world_coord,
            normal,
        }
    }

    fn fragment(&self, varyings: &[FinalRenderVarying; 3], b: BaryCoords) -> Color {
        let (alpha, beta, gamma) = (b.0.x, b.0.y, b.0.z);
        let ambient_intensity = 0.3;
        let (na, nb, nc) = (varyings[0].normal, varyings[1].normal, varyings[2].normal);
        let (wa, wb, wc) = (
            varyings[0].world_coord,
            varyings[1].world_coord,
            varyings[2].world_coord,
        );

        let width = self.light_pov_depths.width();
        let height = self.light_pov_depths.height();

        let object_pos = alpha * wa + beta * wb + gamma * wc;

        let light_dir = (object_pos - Vec4::from((self.light_pos, 1.))).normalize();
        let m_light_to_world = Mat4::IDENTITY;
        let transformed_light_dir = m_light_to_world * light_dir;

        let normal = alpha * na + beta * nb + gamma * nc;
        let dir_intensity = normal.dot(-transformed_light_dir).clamp(0., 1.);
        let dir_intensity = (dir_intensity * 6.).round() / 6.;
        let dir_intensity = dir_intensity * (1. - ambient_intensity);

        let total_intensity = ambient_intensity + dir_intensity;
        let mut color = vec3(255., 155., 0.) * total_intensity;

        let this_pixel_world_coords = alpha * wa + beta * wb + gamma * wc;

        let this_pixel_clip_coords = self.light_vp * this_pixel_world_coords;
        let this_pixel_ndc = perspective_divided(this_pixel_clip_coords);
        let this_pixel_screen_coords = (self.light_viewport * this_pixel_ndc).xyz();
        let p = this_pixel_screen_coords.with_z(this_pixel_screen_coords.z / 2. + 0.5);

        if (p.x as i32) >= 0
            && (p.y as i32) >= 0
            && (p.x as usize) < width
            && (p.y as usize) < height
        {
            let light_pov_best_z = self.light_pov_depths.get(p.x as usize, p.y as usize);
            if p.z < light_pov_best_z + 0.005 {
                // do nothing; we're in light
            } else {
                let total_intensity = ambient_intensity;
                color = vec3(255., 155., 0.) * total_intensity;
            }
        }

        color.as_u8vec3()
    }
}

impl<'buf> FinalRenderShaders<'buf> {
    fn new(
        light_pos: Vec3,
        light_vp: Mat4,
        light_viewport: Mat4,
        light_pov_depths: &'buf DepthBuffer,
    ) -> Self {
        Self {
            light_pos,
            light_vp,
            light_viewport,
            light_pov_depths,
        }
    }
}

/*
We want there to be per-vertex data.
We will interpolate it and pass it into the fragment shader
OR
we could pass in the barycentric coords. And let the frag shader do it.
*/
fn triangle<S>(
    a: Vec3,
    b: Vec3,
    c: Vec3,
    varyings: &[S::Varying; 3],
    shader: &S,
    image: &mut Option<&mut Image>,
    depths: &mut DepthBuffer,
) -> RenderingResult
where
    S: Shader,
{
    let mut answer = RenderingResult::new();
    answer.num_unclipped_triangles_considered += 1;

    let width = depths.width();
    let height = depths.height();

    let smallest_x = f32::min(a.x, f32::min(b.x, c.x)) as i32;
    let smallest_y = f32::min(a.y, f32::min(b.y, c.y)) as i32;
    let biggest_x = f32::max(a.x, f32::max(b.x, c.x)) as i32;
    let biggest_y = f32::max(a.y, f32::max(b.y, c.y)) as i32;

    let smallest_x = i32::max(smallest_x, 0);
    let smallest_y = i32::max(smallest_y, 0);
    let biggest_x = i32::min(biggest_x, width as i32 - 1);
    let biggest_y = i32::min(biggest_y, height as i32 - 1);

    if smallest_x > biggest_x || smallest_y > biggest_y {
        return answer;
    }

    answer.num_triangles_with_onscreen_bb += 1;
    let total_area = signed_triangle_area(a, b, c);
    if total_area <= 0. {
        return answer;
    }

    for x in smallest_x..=biggest_x {
        for y in smallest_y..=biggest_y {
            answer.num_bb_pixels_considered += 1;
            let p = Vec3::new(x as f32, y as f32, 0.);

            let alpha = signed_triangle_area(p, b, c) / total_area;
            if alpha < 0.0 {
                continue;
            }

            let beta = signed_triangle_area(p, c, a) / total_area;
            if beta < 0.0 {
                continue;
            }

            let gamma = signed_triangle_area(p, a, b) / total_area;
            if gamma < 0.0 {
                continue;
            }

            let z = alpha * a.z + beta * b.z + gamma * c.z;
            let z = z / 2. + 0.5;
            // assert!(z >= 0.);
            // assert!(z <= 1.);
            answer.num_triangle_pixels_considered += 1;
            if x >= 0 && x < width as i32 && y >= 0 && y < height as i32 {
                answer.num_in_bounds_triangle_pixels_considered += 1;
                let x = x as usize;
                let y = y as usize;
                if z < depths.get(x, y) {
                    depths.set(x, y, z);
                    answer.num_depth_buffer_sets += 1;

                    if let Some(image) = image {
                        let color = shader.fragment(varyings, BaryCoords(vec3(alpha, beta, gamma)));
                        image.set(x, y, color);
                        answer.num_pixels_drawn += 1
                    }
                }
            }
        }
    }

    answer
}

fn should_clip(clip_coordinates: &Vec4) -> bool {
    let w = clip_coordinates.w;
    if clip_coordinates.x < -w
        || clip_coordinates.x > w
        || clip_coordinates.y < -w
        || clip_coordinates.y > w
        || clip_coordinates.z < -w
        || clip_coordinates.z > w
    {
        return true;
    }

    false
}

impl World {
    fn new(args: &Args) -> Self {
        let mut objects = Vec::new();

        let grid_size = 1;
        let mut idx = 0;
        for i in 0..grid_size {
            for j in 0..grid_size {
                for k in 0..grid_size {
                    // if idx as f32 >= num_models {
                    //     continue;
                    // }
                    let model_filename = &args.models[idx % args.models.len()];
                    idx += 1;

                    let mut model =
                        wavefront_obj::Mesh::from_file(model_filename.as_str()).unwrap();

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
                        pos: vec3(i as f32 * 2., j as f32 * 3., k as f32 * 4.),
                        angle_x: 0.,
                        angle_y: 0.,
                        scale: 1.,
                    };

                    objects.push(object);
                }
            }
        }
        for i in 0..10 {
            for j in 0..10 {
                objects.push(Object {
                    mesh: Mesh::wall(),
                    pos: objects[0].pos + vec3(-5. + i as f32, -5. + j as f32, -5.),
                    angle_x: 0.,
                    angle_y: 0.,
                    scale: 1.,
                });

                objects.push(Object {
                    mesh: Mesh::wall(),
                    pos: objects[0].pos + vec3(-4. + i as f32, -5. + j as f32, 4.),
                    angle_x: 0.,
                    angle_y: 180f32.to_radians(),
                    scale: 1.,
                });

                objects.push(Object {
                    mesh: Mesh::wall(),
                    pos: objects[0].pos + vec3(4., -5. + j as f32, -5. + i as f32),
                    angle_x: 0.,
                    angle_y: -90f32.to_radians(),
                    scale: 1.,
                });

                objects.push(Object {
                    mesh: Mesh::wall(),
                    pos: objects[0].pos + vec3(-5. + i as f32, -5., -4. + j as f32),
                    angle_x: -90f32.to_radians(),
                    angle_y: 0.,
                    scale: 1.,
                });
            }
        }

        // objects[0].scale = 3.;

        let image = Image::new(args.canvas_size, args.canvas_size);
        let depths = DepthBuffer::new(args.canvas_size, args.canvas_size);
        let light_depths = DepthBuffer::new(args.canvas_size, args.canvas_size);

        let initial_camera_pos = vec3(0., 0., 3.);
        let camera_dir = (objects[0].pos - initial_camera_pos).normalize();

        let light = Spotlight {
            pos: initial_camera_pos,
            dir: camera_dir,
        };

        Self {
            image,
            depths,
            light_depths,
            objects,
            width: args.canvas_size as usize,
            render_settings: RenderSettings {
                wireframe: false,
                no_triangles: false,
                draw_lightbulb: false,
            },
            camera: Camera {
                pos: initial_camera_pos,
                dir: camera_dir,
                up: vec3(0., 1., 0.).normalize(),
            },
            keys: HashSet::new(),
            first_pressed_this_frame: HashSet::new(),
            light,
            time_since_start: Duration::from_secs(0),
            angle_time: Duration::from_secs(0),
            movement_settings: MovementSettings {
                rotate_objects: false,
                move_light_around: false,
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

    fn is_key_down(&self, key: KeyCode) -> bool {
        self.keys.contains(&key)
    }

    fn was_key_pressed(&self, key: KeyCode) -> bool {
        self.first_pressed_this_frame.contains(&key)
    }

    fn update(&mut self, since_last_frame: Duration, since_start: Duration) {
        if self.is_key_down(KeyCode::KeyW) {
            self.move_(Direction::Forward);
        }

        if self.is_key_down(KeyCode::KeyS) {
            self.move_(Direction::Back);
        }

        if self.is_key_down(KeyCode::KeyA) {
            self.move_(Direction::Left);
        }

        if self.is_key_down(KeyCode::KeyD) {
            self.move_(Direction::Right);
        }
        if self.was_key_pressed(KeyCode::Digit1) {
            self.render_settings.wireframe = !self.render_settings.wireframe;
        }
        if self.was_key_pressed(KeyCode::Digit2) {
            self.render_settings.no_triangles = !self.render_settings.no_triangles;
        }
        if self.was_key_pressed(KeyCode::Digit3) {
            self.movement_settings.move_light_around = !self.movement_settings.move_light_around;
        }
        if self.was_key_pressed(KeyCode::Digit4) {
            self.movement_settings.rotate_objects = !self.movement_settings.rotate_objects;
        }
        if self.was_key_pressed(KeyCode::Digit5) {
            self.render_settings.draw_lightbulb = !self.render_settings.draw_lightbulb;
        }

        self.time_since_start = since_start;

        if self.movement_settings.rotate_objects {
            self.angle_time += since_last_frame;
        }

        for (idx, object) in self.objects.iter_mut().enumerate() {
            if object.mesh.num_faces() > 10 {
                let angle = self.angle_time.as_secs_f32() * (idx as f32 + 1.);

                object.angle_y = angle;
            }
        }

        let t = self.time_since_start.as_secs_f32();

        self.light.pos =
            self.camera.pos + vec3(0.1 * f32::sin(t), -0.6 + 0.1 * f32::cos(0.7 * t), -0.1);
        self.light.dir = self.camera.dir;

        self.first_pressed_this_frame.clear();
    }

    fn render_object<S>(
        object: &Object,
        uniforms: &RenderingUniforms,
        image: &mut Option<&mut Image>,
        depths: &mut DepthBuffer,
        render_settings: &RenderSettings,
        shader: &S,
    ) -> RenderingResult
    where
        S: Shader,
    {
        let mut answer = RenderingResult::new();

        let RenderingUniforms {
            m_viewport,
            m_projection,
            m_view,
        } = uniforms;

        let m_scale = Mat4::from_scale(vec3(object.scale, object.scale, object.scale));
        let m_rot = Mat4::from_rotation_y(object.angle_y) * Mat4::from_rotation_x(object.angle_x);
        let m_trans = Mat4::from_translation(object.pos);
        let m_model = m_trans * m_rot * m_scale;

        let m_mvp = m_projection * m_view * m_model;

        {
            let mut clipped_verts = 0;
            for bb_vert in object.mesh.bounding_box_coords() {
                let clip_coordinates = m_mvp * *bb_vert;
                if should_clip(&clip_coordinates) {
                    clipped_verts += 1;
                }
            }

            if clipped_verts == 8 {
                answer.num_objects_bb_culled += 1;
                return answer;
            }
        }

        let m_normal = (m_trans * m_rot).inverse().transpose();

        for face_idx in 0..object.mesh.num_faces() {
            let mut screen_coords: [Vec3; 3] = [Vec3::new(0., 0., 0.); 3];
            let mut world_coords: [Vec4; 3] = [Vec4::ZERO; 3];
            let mut varyings: Vec<S::Varying> = Vec::new(); // TODO make this an array
            let mut clipped_verts: i32 = 0;

            for j in 0..3 {
                let model_coordinates = Vec4::from((object.mesh.vertex(face_idx, j), 1.0));
                world_coords[j] = m_model * model_coordinates;

                let clip_coordinates = m_mvp * model_coordinates;

                if should_clip(&clip_coordinates) {
                    clipped_verts += 1;
                }

                let normalized_device_coordinates = perspective_divided(clip_coordinates);
                // assert!(normalized_device_coordinates.z >= -1.);
                // assert!(normalized_device_coordinates.z <= 1.);

                screen_coords[j] = (m_viewport * normalized_device_coordinates).xyz();
            }

            if clipped_verts == 3 {
                continue;
            }

            if !render_settings.no_triangles {
                // We're going to do lighting by dot-producting the light direction
                // and normals, so it's really THOSE two that need to be transformed
                // with respect to each other. It's also very important that we
                // not normalize or xyz the normals and lighting vectors! Those are
                // non-linear transforms and break the proof that transforming by
                // the transpose of the inverse preserves dot products.
                for i in 0..3 {
                    let normal = object.mesh.normal(face_idx, i);
                    let normal = m_normal * Vec4::from((normal, 0.));
                    varyings.push(shader.vertex(world_coords[i], normal));
                }

                let triangle_result = triangle(
                    screen_coords[0],
                    screen_coords[1],
                    screen_coords[2],
                    &[varyings[0], varyings[1], varyings[2]],
                    shader,
                    image,
                    depths,
                );

                answer = answer + triangle_result;
            }

            if let Some(image) = image
                && render_settings.wireframe
            {
                for i in 0..3 {
                    let line_result = linevf32(
                        screen_coords[i % 3].xy(),
                        screen_coords[(i + 1) % 3].xy(),
                        image,
                        RED,
                    );

                    answer = answer + line_result;
                }
            }
        }

        answer
    }

    fn render(&mut self) -> RenderingResult {
        let mut answer = RenderingResult::new();

        let canvas_size = self.width as f32;

        let z_near = 1.;
        let z_far = 50.;
        let m_projection_light = Mat4::perspective_rh_gl(f32::to_radians(70.), 1.0, z_near, z_far);

        let m_viewport = Mat4::from_scale(Vec3::new(canvas_size / 2.0, canvas_size / 2.0, 1.))
            * Mat4::from_translation(Vec3::new(1.0, 1.0, 0.0));

        let light_pov = NoopShaderColorsWhite::new();

        // First from the light's POV
        let m_light_view = Mat4::look_to_rh(self.light.pos, self.light.dir, self.camera.up);

        let light_uniforms = RenderingUniforms {
            m_viewport,
            m_projection: m_projection_light,
            m_view: m_light_view,
        };
        for object in self.objects.iter() {
            let light_pov_rendering_result = Self::render_object(
                object,
                &light_uniforms,
                &mut None,
                &mut self.light_depths,
                &self.render_settings,
                &light_pov,
            );
            answer = answer + light_pov_rendering_result;
        }

        let m_projection_objects =
            Mat4::perspective_rh_gl(f32::to_radians(60.), 1.0, z_near, z_far);

        let m_view = Mat4::look_to_rh(self.camera.pos, self.camera.dir, self.camera.up);
        let uniforms = RenderingUniforms {
            m_viewport,
            m_projection: m_projection_objects,
            m_view,
        };

        // if self.render_settings.draw_lightbulb {
        //     let lightbulb_shader = NoopShaderColorsWhite::new();
        //     answer = answer
        //         + Self::render_object(
        //             &self.light,
        //             &uniforms,
        //             &mut Some(&mut self.image),
        //             &mut self.depths,
        //             &self.render_settings,
        //             &|w, n| lightbulb_shader.vertex_shader(w, n),
        //             &|v, b| lightbulb_shader.fragment_shader(v, b),
        //         );
        // }

        // Now the final render
        let final_render = FinalRenderShaders::new(
            self.light.pos,
            light_uniforms.m_projection * light_uniforms.m_view,
            light_uniforms.m_viewport,
            &self.light_depths,
        );

        for object in self.objects.iter() {
            answer = answer
                + Self::render_object(
                    object,
                    &uniforms,
                    &mut Some(&mut self.image),
                    &mut self.depths,
                    &self.render_settings,
                    &final_render,
                );
        }

        answer
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

    fn draw(&mut self, frame: &mut [u8]) -> RenderingResult {
        self.clear();
        let rendering_result = self.render();

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

        rendering_result
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
                self.total_frames += 1;

                self.last_frame = Instant::now();
                let rendering_result = self.world.draw(self.pixels.as_mut().unwrap().frame_mut());
                if self.total_frames % 60 == 0 {
                    log::info!(
                        "{:?} average fps {}, this frame {}",
                        rendering_result,
                        average_fps,
                        this_frame_fps
                    );
                }
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

fn linei32(ax: i32, ay: i32, bx: i32, by: i32, image: &mut Image, color: Color) -> RenderingResult {
    let mut answer = RenderingResult::new();

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
            answer.num_pixels_drawn += 1;
            image.set(xx as usize, yy as usize, color);
        }

        ierror += (by - ay).abs() * 2;
        let should_incr = (ierror > (bx - ax)) as i32;
        y += dy * should_incr;
        ierror -= 2 * (bx - ax) * should_incr;
        x += 1;
    }

    answer
}

fn linef32(ax: f32, ay: f32, bx: f32, by: f32, image: &mut Image, color: Color) -> RenderingResult {
    linei32(ax as i32, ay as i32, bx as i32, by as i32, image, color)
}

fn linevf32(a: Vec2, b: Vec2, image: &mut Image, color: Color) -> RenderingResult {
    linef32(a.x, a.y, b.x, b.y, image, color)
}

#[inline]
fn signed_triangle_area(a: Vec3, b: Vec3, c: Vec3) -> f32 {
    let answer = (b.y - a.y) * (b.x + a.x) + (c.y - b.y) * (c.x + b.x) + (a.y - c.y) * (a.x + c.x);
    0.5 * answer
}

fn perspective_divided(v: Vec4) -> Vec4 {
    v / v.w
}

#[derive(Parser)]
struct Args {
    /// Model files
    models: Vec<String>,

    /// Output image size in pixels. We only do square images for now.
    #[arg(short, long, default_value_t = 320)]
    canvas_size: u16,
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

    Ok(())
}
