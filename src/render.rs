use crate::game::{Camera, Object, ObjectKind, Spotlight};
use crate::image::*;

use glam::{Mat4, Vec3, Vec3Swizzles, Vec4, Vec4Swizzles, vec3, vec4};
use std::f32;
use std::ops::Add;

mod line;
mod shaderutils;

use self::shaderutils::*;

#[derive(Copy, Clone)]
pub struct Colorf(pub Vec3);

pub enum SplitScreenMode {
    Normal,
    Split,
}

pub enum WireframeMode {
    TrianglesOnly,
    WireframeOnly,
    TrianglesAndWireframes,
}

pub struct RenderSettings {
    pub wireframe: WireframeMode,
    pub split_screen_mode: SplitScreenMode,
}

pub struct Renderer {
    pub render_settings: RenderSettings,
    image: Image,
    depths: DepthBuffer,
    light_depths: DepthBuffer,
    pub width: usize,
    debug_lines: Vec<(Vec3, Vec3, Color)>,
}

impl Renderer {
    pub fn new(canvas_size: u16) -> Self {
        let image = Image::new(canvas_size, canvas_size);
        let depths = DepthBuffer::new(canvas_size, canvas_size);
        let light_depths = DepthBuffer::new(canvas_size, canvas_size);

        Self {
            image,
            depths,
            light_depths,
            width: canvas_size as usize,
            render_settings: RenderSettings {
                split_screen_mode: SplitScreenMode::Normal,
                wireframe: WireframeMode::TrianglesOnly,
            },
            debug_lines: Vec::new(),
        }
    }

    fn render_object<S>(
        object_idx: usize,
        object: &Object,
        uniforms: &RenderingUniforms,
        mut image: Option<&mut Image>,
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
            if render_settings.wireframe.should_render_triangles() {
                let varyings: [S::Varying; 3] = core::array::from_fn(|i| {
                    // We're going to do lighting by dot-producting the light direction
                    // and normals, so it's really THOSE two that need to be transformed
                    // with respect to each other. It's also very important that we
                    // not normalize or xyz the normals and lighting vectors! Those are
                    // non-linear transforms and break the proof that transforming by
                    // the transpose of the inverse preserves dot products.
                    let normal = object.mesh.normal(face_idx, i);
                    let normal = m_normal * Vec4::from((normal, 0.));

                    shader.vertex(object_idx, world_coords[i], normal)
                });

                let triangle_result = triangle(
                    &screen_coords,
                    object_idx,
                    &[varyings[0], varyings[1], varyings[2]],
                    shader,
                    image.as_deref_mut(),
                    depths,
                );

                answer = answer + triangle_result;
            }

            if let Some(image) = &mut image
                && render_settings.wireframe.should_render_wireframe()
            {
                for i in 0..3 {
                    let line_result = line::linevf32(
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
    pub fn debug_draw_line_in_world_space(&mut self, a: Vec3, b: Vec3, color: Color) {
        self.debug_lines.push((a, b, color))
    }

    // TODO make this different depending on topdown, FPS and shadow camera
    //
    // for shadow camera we want the near clipping plane further away so we can
    // have more precision over the whole range
    fn clipping_planes() -> (f32, f32) {
        (1., 25.)
    }

    fn render_debug_lines(
        self_width: usize,
        image: &mut Image,
        camera: &Camera,
        a: Vec3,
        b: Vec3,
        color: Color,
    ) {
        let canvas_size = self_width as f32;
        let (z_near, z_far) = Self::clipping_planes();
        let m_projection = Mat4::perspective_rh_gl(f32::to_radians(60.), 1.0, z_near, z_far);

        let m_viewport = Mat4::from_scale(Vec3::new(canvas_size / 2.0, canvas_size / 2.0, 1.))
            * Mat4::from_translation(Vec3::new(1.0, 1.0, 0.0));
        let m_view = Mat4::look_to_rh(camera.pos, camera.dir, camera.up);
        let m_model = Mat4::IDENTITY; // points are given in world space

        let m_mvp = m_projection * m_view * m_model;

        let mut screen_coords: [Vec3; 2] = [Vec3::new(0., 0., 0.); 2];

        for (idx, model_coordinates) in [a, b].iter().enumerate() {
            let clip_coordinates = m_mvp * Vec4::from((*model_coordinates, 1.0));

            let normalized_device_coordinates = perspective_divided(clip_coordinates);

            screen_coords[idx] = (m_viewport * normalized_device_coordinates).xyz();
        }

        let [screen_a, screen_b] = screen_coords;

        line::linevf32(screen_a.xy(), screen_b.xy(), image, color);
    }

    fn render(
        &mut self,
        light: &Spotlight,
        camera: &Camera,
        objects: &Vec<Object>,
    ) -> RenderingResult {
        let mut answer = RenderingResult::new();

        let canvas_size = self.width as f32;
        let (z_near, z_far) = Self::clipping_planes();
        let m_projection_light = Mat4::perspective_rh_gl(f32::to_radians(70.), 1.0, z_near, z_far);

        let m_viewport = Mat4::from_scale(Vec3::new(canvas_size / 2.0, canvas_size / 2.0, 1.))
            * Mat4::from_translation(Vec3::new(1.0, 1.0, 0.0));

        let light_pov = NoopShaderColorsWhite::new();

        // First from the light's POV
        let m_light_view = Mat4::look_to_rh(light.pos, light.dir, camera.up);

        let light_uniforms = RenderingUniforms {
            m_viewport,
            m_projection: m_projection_light,
            m_view: m_light_view,
        };
        for (object_idx, object) in objects.iter().enumerate() {
            if object.kind == ObjectKind::Light || !object.visible {
                continue;
            }
            let light_pov_rendering_result = Self::render_object(
                object_idx,
                object,
                &light_uniforms,
                None,
                &mut self.light_depths,
                &self.render_settings,
                &light_pov,
            );
            answer = answer + light_pov_rendering_result;
        }

        let m_projection_objects =
            Mat4::perspective_rh_gl(f32::to_radians(60.), 1.0, z_near, z_far);

        let m_view = Mat4::look_to_rh(camera.pos, camera.dir, camera.up);
        let uniforms = RenderingUniforms {
            m_viewport,
            m_projection: m_projection_objects,
            m_view,
        };

        // Now the final render
        let final_render = FinalRenderShader::new(
            *light,
            light_uniforms.m_projection * light_uniforms.m_view,
            light_uniforms.m_viewport,
            &self.light_depths,
            objects,
        );

        for (object_idx, object) in objects.iter().enumerate() {
            if !object.visible {
                continue;
            }
            let render_result = match object.kind {
                ObjectKind::Light => Self::render_object(
                    object_idx,
                    object,
                    &uniforms,
                    Some(&mut self.image),
                    &mut self.depths,
                    &self.render_settings,
                    &light_pov,
                ),
                ObjectKind::Exhibit | ObjectKind::WallOrFloor => Self::render_object(
                    object_idx,
                    object,
                    &uniforms,
                    Some(&mut self.image),
                    &mut self.depths,
                    &self.render_settings,
                    &final_render,
                ),
            };

            answer = answer + render_result;
        }

        for (a, b, color) in self.debug_lines.iter() {
            Self::render_debug_lines(self.width, &mut self.image, camera, *a, *b, *color);
        }
        answer
    }

    fn clear(&mut self) {
        self.image.clear(coloru8(0x00, 0x00, 0x00));
        self.depths.clear();
        self.light_depths.clear();
    }

    pub fn draw(
        &mut self,
        light: &Spotlight,
        camera: &Camera,
        objects: &Vec<Object>,
        frame: &mut [u8],
    ) -> RenderingResult {
        self.clear();
        let rendering_result = self.render(light, camera, objects);
        self.debug_lines.clear();

        assert!(self.image.width() == self.width);
        assert!(self.image.height() == self.width);
        let image_buf = self.image.buf().as_slice();

        match self.render_settings.split_screen_mode {
            SplitScreenMode::Normal => {
                // This is the hot path!
                let image_slice = unsafe {
                    std::slice::from_raw_parts(image_buf.as_ptr() as *const u32, image_buf.len())
                };
                let frame_slice = unsafe {
                    std::slice::from_raw_parts_mut(frame.as_mut_ptr() as *mut u32, image_buf.len())
                };

                frame_slice.copy_from_slice(image_slice);
            }
            SplitScreenMode::Split => {
                let split_screen_frame_width = self.width * 2;

                for y in 0..self.width {
                    let image_slice = unsafe {
                        std::slice::from_raw_parts(
                            image_buf[y * self.width..].as_ptr() as *const u32,
                            self.width,
                        )
                    };
                    let frame_slice = unsafe {
                        std::slice::from_raw_parts_mut(
                            frame[4 * y * split_screen_frame_width..].as_ptr() as *mut u32,
                            self.width,
                        )
                    };
                    frame_slice.copy_from_slice(image_slice);
                }
            }
        }

        match self.render_settings.split_screen_mode {
            SplitScreenMode::Normal => {
                // do nothing
            }
            SplitScreenMode::Split => {
                assert!(self.light_depths.width() == self.width);
                assert!(self.light_depths.height() == self.width);
                let split_screen_frame_width = self.width * 2;

                let depth_buf = self.light_depths.buf();
                let min_depth = self.light_depths.min_depth();
                let max_depth = self.light_depths.max_depth();

                for x in 0..self.width {
                    for y in 0..self.width {
                        let image_idx = y * self.width + x;
                        let frame_idx = 4 * y * split_screen_frame_width + 4 * (x + self.width);

                        let depth = depth_buf[image_idx];
                        let gray = DepthBuffer::depth_to_u8(depth, min_depth, max_depth);

                        let color = [gray, gray, gray, 255];
                        frame[frame_idx..frame_idx + 4].copy_from_slice(&color);
                    }
                }
            }
        }

        rendering_result
    }
}

impl WireframeMode {
    pub fn next(&self) -> Self {
        match self {
            Self::TrianglesOnly => Self::WireframeOnly,
            Self::WireframeOnly => Self::TrianglesAndWireframes,
            Self::TrianglesAndWireframes => Self::TrianglesOnly,
        }
    }

    fn should_render_triangles(&self) -> bool {
        match self {
            Self::TrianglesOnly => true,
            Self::WireframeOnly => false,
            Self::TrianglesAndWireframes => true,
        }
    }

    fn should_render_wireframe(&self) -> bool {
        match self {
            Self::TrianglesOnly => false,
            Self::WireframeOnly => true,
            Self::TrianglesAndWireframes => true,
        }
    }
}

struct RenderingUniforms {
    m_viewport: Mat4,
    m_projection: Mat4,
    m_view: Mat4,
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub struct RenderingResult {
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

    fn vertex(&self, object_idx: usize, coord: Vec4, normal: Vec4) -> Self::Varying;
    fn fragment(&self, object_idx: usize, varyings: &[Self::Varying; 3], b: BaryCoords) -> Color;
}

struct NoopShaderColorsWhite {
    //
}

impl Shader for NoopShaderColorsWhite {
    type Varying = ();

    #[inline]
    fn vertex(&self, _: usize, _coord: Vec4, _normal: Vec4) {}

    #[inline]
    fn fragment(&self, _: usize, _varyings: &[(); 3], _b: BaryCoords) -> Color {
        coloru8(255, 255, 255)
    }
}

impl NoopShaderColorsWhite {
    fn new() -> Self {
        Self {}
    }
}

struct FinalRenderShader<'buf> {
    spotlight: Spotlight,
    light_vp: Mat4,
    light_viewport: Mat4,
    light_pov_depths: &'buf DepthBuffer,

    objects: &'buf Vec<Object>,
}

#[derive(Clone, Copy)]
struct FinalRenderVarying {
    world_coord: Vec4,
    normal: Vec4,
}

impl<'buf> Shader for FinalRenderShader<'buf> {
    type Varying = FinalRenderVarying;

    fn vertex(&self, _object_idx: usize, world_coord: Vec4, normal: Vec4) -> FinalRenderVarying {
        FinalRenderVarying {
            world_coord,
            normal,
        }
    }

    fn fragment(
        &self,
        object_idx: usize,
        varyings: &[FinalRenderVarying; 3],
        b: BaryCoords,
    ) -> Color {
        let constant_dir_light = vec4(0.1, -0.2, 0.3, 0.).normalize();
        let ambient_factor = 0.1;
        let dir_factor = 0.2;
        let flashlight_factor = 0.7;

        let (alpha, beta, gamma) = (b.0.x, b.0.y, b.0.z);
        let (na, nb, nc) = (varyings[0].normal, varyings[1].normal, varyings[2].normal);
        let (wa, wb, wc) = (
            varyings[0].world_coord,
            varyings[1].world_coord,
            varyings[2].world_coord,
        );

        let width = self.light_pov_depths.width();
        let height = self.light_pov_depths.height();

        let this_pixel_world_coords = alpha * wa + beta * wb + gamma * wc;

        let light_to_pixel = this_pixel_world_coords - Vec4::from((self.spotlight.pos, 1.));
        let light_to_pixel_distance = light_to_pixel.length();
        let light_to_pixel_normalized = light_to_pixel / light_to_pixel_distance;

        fn assert_normalized(x: Vec4) {
            assert!((x.length() - 1.).abs() < 1e-6)
        }
        assert_normalized(light_to_pixel_normalized);
        // light_dir is in the world coordinates, so we don't need to transform it.
        let light_dir = Vec4::from((self.spotlight.dir, 0.)).normalize();
        assert_normalized(light_dir);

        let spotlight_factor = light_to_pixel_normalized.dot(light_dir);
        let spotlight_factor_hue = (1. - smoothstep(0.934, 0.936, spotlight_factor)) * 2. / 3.;
        let spotlight_factor_lightness = smoothstep(0.93, 0.94, spotlight_factor);
        let spotlight_saturation = 0.3;
        let color_spotlight_factor = hsl2rgb(
            spotlight_factor_hue,
            spotlight_saturation,
            spotlight_factor_lightness,
        );
        let distance_factor = 1. - smoothstep(10., 15., light_to_pixel_distance);

        let this_pixel_normal = alpha * na + beta * nb + gamma * nc;
        let flashlight_intensity = this_pixel_normal.dot(-light_dir).clamp(0., 1.);
        let flashlight_intensity = flashlight_intensity * color_spotlight_factor * distance_factor;

        let dir_intensity = this_pixel_normal.dot(-constant_dir_light).clamp(0., 1.);

        let this_pixel_clip_coords = self.light_vp * this_pixel_world_coords;
        let this_pixel_ndc = perspective_divided(this_pixel_clip_coords);
        let this_pixel_screen_coords = (self.light_viewport * this_pixel_ndc).xyz();
        let p = this_pixel_screen_coords.with_z(this_pixel_screen_coords.z / 2. + 0.5);

        let mut flashlight_intensity = flashlight_intensity;
        if (p.x as i32) >= 0
            && (p.y as i32) >= 0
            && (p.x as usize) < width
            && (p.y as usize) < height
        {
            let light_pov_best_z = self.light_pov_depths.get(p.x as usize, p.y as usize);
            if p.z < light_pov_best_z + 0.005 {
                // TODO move all the spotlight computation here.
                // do nothing; we're in light
            } else {
                // let total_intensity = ambient_intensity + dir_intensity * (1. - ambient_intensity);
                flashlight_intensity = Vec3::ZERO;
            }
        }

        let total_intensity = 1. * ambient_factor
            + flashlight_intensity * flashlight_factor
            + dir_intensity * dir_factor;

        let object_color = self.objects[object_idx].color.0;

        let color = object_color * total_intensity;
        colorvf(color * 255.)
    }
}

impl<'buf> FinalRenderShader<'buf> {
    fn new(
        spotlight: Spotlight,
        light_vp: Mat4,
        light_viewport: Mat4,
        light_pov_depths: &'buf DepthBuffer,
        objects: &'buf Vec<Object>,
    ) -> Self {
        Self {
            spotlight,
            light_vp,
            light_viewport,
            light_pov_depths,
            objects,
        }
    }
}

fn triangle<S>(
    verts: &[Vec3; 3],
    object_idx: usize,
    varyings: &[S::Varying; 3],
    shader: &S,
    mut image: Option<&mut Image>,
    depths: &mut DepthBuffer,
) -> RenderingResult
where
    S: Shader,
{
    let mut answer = RenderingResult::new();
    answer.num_unclipped_triangles_considered += 1;

    let width = depths.width();
    let height = depths.height();

    let [a, b, c] = verts;

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

            let alpha = signed_triangle_area(&p, b, c) / total_area;
            if alpha < 0.0 {
                continue;
            }

            let beta = signed_triangle_area(&p, c, a) / total_area;
            if beta < 0.0 {
                continue;
            }

            let gamma = signed_triangle_area(&p, a, b) / total_area;
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

                    if let Some(image) = &mut image {
                        let color = shader.fragment(
                            object_idx,
                            varyings,
                            BaryCoords(vec3(alpha, beta, gamma)),
                        );
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

#[inline]
fn signed_triangle_area(a: &Vec3, b: &Vec3, c: &Vec3) -> f32 {
    let answer = (b.y - a.y) * (b.x + a.x) + (c.y - b.y) * (c.x + b.x) + (a.y - c.y) * (a.x + c.x);
    0.5 * answer
}

fn perspective_divided(v: Vec4) -> Vec4 {
    v / v.w
}
