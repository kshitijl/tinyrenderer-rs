mod image;
mod linalg;
mod tga;
mod wavefront_obj;

use crate::image::*;
use crate::linalg::*;
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

fn fill_triangle(
    ax: i32,
    ay: i32,
    bx: i32,
    by: i32,
    cx: i32,
    cy: i32,
    image: &mut Image,
    color: Color,
) {
    assert!(ay <= by);
    assert!(by <= cy);

    let mut y = ay;

    // TODO carefully think through degenerate cases: ay = by, by = cy, and when
    // ay = by = cy. Note that ay <= by <= cy so those are the three cases.
    while y <= cy {
        let x_ab_or_bc = if y < by {
            assert!(by != ay);
            let x_ab = ax + ((y - ay) * (bx - ax)) / (by - ay);
            x_ab
        } else if y < cy {
            assert!(cy != by);
            let x_bc = bx + ((y - by) * (cx - bx)) / (cy - by);
            x_bc
        } else {
            cx
        };

        let x_ac = if ay != cy {
            assert!(cy != ay);
            ax + ((y - ay) * (cx - ax)) / (cy - ay)
        } else {
            ax
        };

        let lower_bound = i32::min(x_ab_or_bc, x_ac);
        let upper_bound = i32::max(x_ab_or_bc, x_ac);

        for x in lower_bound..=upper_bound {
            image.set(x as usize, y as usize, color);
        }

        y += 1;
    }
}

#[inline]
fn same_side(p1: Vec2<i32>, p2: Vec2<i32>, a: Vec2<i32>, b: Vec2<i32>) -> bool {
    let cp1 = (b - a).cross(p1 - a);
    let cp2 = (b - a).cross(p2 - a);
    cp1 * cp2 >= 0
}

fn point_in_triangle(p: Vec2<i32>, a: Vec2<i32>, b: Vec2<i32>, c: Vec2<i32>) -> bool {
    same_side(p, a, b, c) && same_side(p, b, a, c) && same_side(p, c, a, b)
}

fn fill_triangle_pixeltest(
    ax: i32,
    ay: i32,
    bx: i32,
    by: i32,
    cx: i32,
    cy: i32,
    image: &mut Image,
    color: Color,
) {
    let smallest_x = i32::min(ax, i32::min(bx, cx));
    let smallest_y = i32::min(ay, i32::min(by, cy));
    let biggest_x = i32::max(ax, i32::max(bx, cx));
    let biggest_y = i32::max(ay, i32::max(by, cy));

    let a = vec2(ax, ay);
    let b = vec2(bx, by);
    let c = vec2(cx, cy);
    for x in smallest_x..=biggest_x {
        for y in smallest_y..=biggest_y {
            if point_in_triangle(vec2(x, y), a, b, c) {
                image.set(x as usize, y as usize, color);
            }
        }
    }
}
fn trianglei32(
    ax: i32,
    ay: i32,
    bx: i32,
    by: i32,
    cx: i32,
    cy: i32,
    image: &mut Image,
    color: Color,
) {
    let mut arr = [(ax, ay), (bx, by), (cx, cy)];
    arr.sort_unstable_by_key(|p| p.1);
    fill_triangle(
        arr[0].0, arr[0].1, arr[1].0, arr[1].1, arr[2].0, arr[2].1, image, color,
    );
}

fn trianglev2i(a: Vec2<i32>, b: Vec2<i32>, c: Vec2<i32>, image: &mut Image, color: Color) {
    trianglei32(a.x, a.y, b.x, b.y, c.x, c.y, image, color)
}

fn main() -> std::io::Result<()> {
    let (width, height) = (800, 800);
    let animate = true;
    let model = wavefront_obj::Model::from_file("assets/diablo3_pose.obj").unwrap();

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
