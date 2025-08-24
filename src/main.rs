mod image;
mod linalg;
mod tga;
mod wavefront_obj;

use crate::image::*;
use crate::linalg::*;
use std::f32;

fn signed_triangle_area(a: Vec2<i32>, b: Vec2<i32>, c: Vec2<i32>) -> f32 {
    let answer = (b.y - a.y) * (b.x + a.x) + (c.y - b.y) * (c.x + b.x) + (a.y - c.y) * (a.x + c.x);
    0.5 * answer as f32
}

fn triangle(
    a: Vec3<i32>,
    b: Vec3<i32>,
    c: Vec3<i32>,
    image: &mut Image,
    depths: &mut DepthBuffer,
    color: Color,
) {
    let total_area = signed_triangle_area(a.xy(), b.xy(), c.xy());

    let smallest_x = i32::min(a.x, i32::min(b.x, c.x));
    let smallest_y = i32::min(a.y, i32::min(b.y, c.y));
    let biggest_x = i32::max(a.x, i32::max(b.x, c.x));
    let biggest_y = i32::max(a.y, i32::max(b.y, c.y));

    for x in smallest_x..=biggest_x {
        for y in smallest_y..=biggest_y {
            let p = vec2(x, y);

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

            let z = alpha * a.z as f32 + beta * b.z as f32 + gamma * c.z as f32;
            let z = z as u8;

            if x >= 0 && x < image.width() as i32 && y >= 0 && y < image.height() as i32 {
                if depths.get(x as usize, y as usize) < z {
                    depths.set(x as usize, y as usize, z);
                    image.set(x as usize, y as usize, color);
                }
            }
        }
    }
}

fn main() -> std::io::Result<()> {
    let s = 800;
    let animate = false;
    let model = wavefront_obj::Model::from_file("assets/head.obj").unwrap();

    let final_angle = if animate { 360 } else { 1 };
    for (idx, angle) in (0..final_angle).step_by(20).enumerate() {
        let mut image = Image::new(s, s);
        let mut depths = DepthBuffer::new(s, s);
        let s = s as f32;

        let angle = angle as f32 * 2.0 * f32::consts::PI / 360.0;
        // let cos_angle = angle.cos();
        let sin_angle = angle.sin();
        let light_dir = vec3(sin_angle, 0.0, -1.0).normalized();

        for face in model.faces.iter() {
            let mut screen_coords: [Vec3<i32>; 3] = [vec3(0, 0, 0); 3];
            let mut world_coords: [Vec3<f32>; 3] = [vec3(0.0, 0.0, 0.0); 3];

            for j in 0..3 {
                let v = model.vertices[face[j]];
                screen_coords[j] = vec3(
                    ((v.x + 1.) * s / 2.0) as i32,
                    ((v.y + 1.) * s / 2.0) as i32,
                    ((v.z + 1.) * 255. / 2.0) as i32,
                );
                world_coords[j] = v;
            }

            let normal = ((world_coords[2] - world_coords[0])
                .cross(world_coords[1] - world_coords[0]))
            .normalized();

            let intensity = normal.dot(light_dir);

            if intensity > 0.0 {
                let gray = (intensity * 255.0) as u8;
                let triangle_color = color(gray, gray, gray);

                triangle(
                    screen_coords[0],
                    screen_coords[1],
                    screen_coords[2],
                    &mut image,
                    &mut depths,
                    triangle_color,
                );
            }
        }
        let tga = tga::TgaFile::from_image(image);
        tga.save_to_path(format!("raw-output/frame-{:02}.tga", idx).as_str())?;

        let depth_tga = tga::TgaFile::from_image(depths.to_image());
        depth_tga.save_to_path(format!("raw-output/depth-{:02}.tga", idx).as_str())?;
    }

    Ok(())
}
