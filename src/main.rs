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
use glam::{Mat4, Vec2, Vec3, Vec3Swizzles, Vec4, Vec4Swizzles, vec3};
use std::f32;

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
            assert!(z >= 0.);
            assert!(z <= 1.);

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
                    let color = coloru8(color.x, color.y, color.z);

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

    /// Produce many frames, not just one
    #[arg(short, long)]
    animate: bool,

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
    let args = Args::parse();
    let model = wavefront_obj::Model::from_file(args.model.as_str()).unwrap();

    println!(
        "Parsed {} vertices, {} faces, {} normals",
        model.num_vertices(),
        model.num_faces(),
        model.num_normals()
    );

    let final_angle = if args.animate { 360 } else { 1 };
    for (idx, angle) in (0..final_angle).step_by(8).enumerate() {
        let mut image = Image::new(args.canvas_size, args.canvas_size);
        let mut depths = DepthBuffer::new(args.canvas_size, args.canvas_size);
        let canvas_size = args.canvas_size as f32;

        let angle = -20;
        let angle = (angle as f32).to_radians();
        let m_rot = Mat4::from_rotation_y(angle) * Mat4::from_rotation_x(5.0f32.to_radians());
        let m_trans = Mat4::from_translation(Vec3::new(0., -0.5, -2.5));
        let m_model = m_trans * m_rot;

        let m_view = Mat4::IDENTITY;

        let z_near = 1.;
        let z_far = 10.;
        let m_projection = Mat4::perspective_rh(f32::to_radians(80.), 1.0, z_near, z_far);

        let m_mvp = m_projection * m_view * m_model;
        let m_mvpit = m_mvp.inverse().transpose();

        let m_viewport = Mat4::from_scale(Vec3::new(canvas_size / 2.0, canvas_size / 2.0, 1.))
            * Mat4::from_translation(Vec3::new(1.0, 1.0, 0.0));

        let light_dir = Vec3::new(-1., -1., -1.).normalize();

        for face_idx in 0..model.num_faces() {
            let mut screen_coords: [Vec3; 3] = [Vec3::new(0., 0., 0.); 3];
            let mut world_coords: [Vec3; 3] = [Vec3::new(0.0, 0.0, 0.0); 3];

            for j in 0..3 {
                let model_coordinates = Vec4::from((model.vertex(face_idx, j), 1.0));

                let world_coordinates = m_model * model_coordinates;

                assert!(world_coordinates.z < 0.);
                assert!(world_coordinates.w == 1.0);

                let eye_coordinates = &m_view * &world_coordinates;

                let clip_coordinates = &m_projection * &eye_coordinates;

                let clip_coordinates = m_mvp * model_coordinates;

                let normalized_device_coordinates = perspective_divided(clip_coordinates);
                assert!(normalized_device_coordinates.z >= -1.);
                assert!(normalized_device_coordinates.z <= 1.);

                screen_coords[j] = (&m_viewport * &normalized_device_coordinates).xyz();
                world_coords[j] = normalized_device_coordinates.xyz();
            }

            // TODO transform the normal properly
            // TODO use all 3 normals, not just this one
            // TODO try toon and gouraud shading
            let mut normals = Vec::new();
            for i in 0..3 {
                let normal = model.normal(face_idx, i);
                let normal = normal.with_z(-normal.z);
                let normal = m_mvpit * Vec4::from((normal, 0.));
                let normal = -normal.xyz().normalize();
                normals.push(normal);
            }

            // let normal = ((world_coords[0] - world_coords[2])
            //     .cross(world_coords[1] - world_coords[0]))
            // .normalize();

            // let intensity = normal.dot(light_dir);

            // let gray = (intensity * 255.0).clamp(0.0, 255.0) as u8;
            // let triangle_color = coloru8(gray, gray, gray);
            // let triangle_color = WHITE;

            triangle(
                screen_coords[0],
                screen_coords[1],
                screen_coords[2],
                light_dir,
                normals[0],
                normals[1],
                normals[2],
                &mut image,
                &mut depths,
            );

            if args.wireframe {
                for i in 0..3 {
                    linevf32(
                        screen_coords[i % 3].xy(),
                        screen_coords[(i + 1) % 3].xy(),
                        &mut image,
                        RED,
                    );
                }
            }
        }
        let tga = tga::TgaFile::from_image(image);
        tga.save_to_path(format!("raw-output/frame-{:02}.tga", idx).as_str())?;

        if args.write_depth_buffer {
            let depth_tga = tga::TgaFile::from_image(depths.to_image());
            depth_tga.save_to_path(format!("raw-output/depth-{:02}.tga", idx).as_str())?;
        }
    }

    Ok(())
}
