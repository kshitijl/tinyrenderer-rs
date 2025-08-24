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

fn fill_triangle_pixeltest(
    a: Vec2<i32>,
    b: Vec2<i32>,
    c: Vec2<i32>,
    image: &mut Image,
    color: Color,
) {
    let total_area = signed_triangle_area(a, b, c);

    let smallest_x = i32::min(a.x, i32::min(b.x, c.x));
    let smallest_y = i32::min(a.y, i32::min(b.y, c.y));
    let biggest_x = i32::max(a.x, i32::max(b.x, c.x));
    let biggest_y = i32::max(a.y, i32::max(b.y, c.y));

    for x in smallest_x..=biggest_x {
        for y in smallest_y..=biggest_y {
            let p = vec2(x, y);

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

            image.set(x as usize, y as usize, color);
        }
    }
}

fn trianglev2i(a: Vec2<i32>, b: Vec2<i32>, c: Vec2<i32>, image: &mut Image, color: Color) {
    // We don't need this sort, but it's nice to keep it around to prevent
    // depending on the order of vertices in the model file.
    let mut arr = [a, b, c];
    arr.sort_unstable_by_key(|p| p.y);
    fill_triangle_pixeltest(arr[0], arr[1], arr[2], image, color);
}

fn main() -> std::io::Result<()> {
    let (width, height) = (800, 800);
    let animate = true;
    let mut model = wavefront_obj::Model::from_file("assets/diablo.obj").unwrap();

    model.faces.sort_unstable_by(|f1, f2| {
        let f1_max = f32::max(
            model.vertices[f1[0]].z,
            f32::max(model.vertices[f1[1]].z, model.vertices[f1[2]].z),
        );
        let f2_max = f32::max(
            model.vertices[f2[0]].z,
            f32::max(model.vertices[f2[1]].z, model.vertices[f2[2]].z),
        );
        f1_max.partial_cmp(&f2_max).unwrap()
    });

    let final_angle = if animate { 360 } else { 1 };
    for (idx, angle) in (0..final_angle).step_by(20).enumerate() {
        let mut image = Image::new(width, height);
        let w = image.width() as f32;
        let h = image.height() as f32;

        let angle = angle as f32 * 2.0 * f32::consts::PI / 360.0;
        // let cos_angle = angle.cos();
        let sin_angle = angle.sin();
        let light_dir = vec3(sin_angle, 0.0, -1.0).normalized();

        for face in model.faces.iter() {
            let mut screen_coords: [Vec2<i32>; 3] = [vec2(0, 0); 3];
            let mut world_coords: [Vec3; 3] = [vec3(0.0, 0.0, 0.0); 3];

            for j in 0..3 {
                let v = model.vertices[face[j]];
                screen_coords[j] = vec2(
                    ((v.x + 1.) * w / 2.0001) as i32,
                    ((v.y + 1.) * h / 2.0001) as i32,
                );
                world_coords[j] = v;
            }

            let normal = ((world_coords[2] - world_coords[0])
                .cross(world_coords[1] - world_coords[0]))
            .normalized();

            let intensity = normal.dot(light_dir);

            // if animate {
            //     a.x = a.x * cos_angle + a.z * sin_angle;
            //     b.x = b.x * cos_angle + b.z * sin_angle;
            //     c.x = c.x * cos_angle + c.z * sin_angle;
            // }

            if intensity > 0.0 {
                let gray = (intensity * 255.0) as u8;
                let triangle_color = color(gray, gray, gray);

                trianglev2i(
                    screen_coords[0],
                    screen_coords[1],
                    screen_coords[2],
                    &mut image,
                    triangle_color,
                );
            }
        }
        let tga = tga::TgaFile::from_image(image);
        tga.save_to_path(format!("raw-output/frame-{:02}.tga", idx).as_str())?;
    }

    Ok(())
}
