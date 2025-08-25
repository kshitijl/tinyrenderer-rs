mod image;
mod linalg;
mod tga;
mod wavefront_obj;

use crate::image::*;
use crate::linalg::*;
use clap::Parser;
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

fn linevi32(a: Vec2<i32>, b: Vec2<i32>, image: &mut Image, color: Color) {
    linei32(a.x, a.y, b.x, b.y, image, color)
}

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

            if x >= 0 && x < image.width() as i32 && y >= 0 && y < image.height() as i32 {
                if depths.get(x as usize, y as usize) > z {
                    depths.set(x as usize, y as usize, z);
                    // let z = alpha * a.z as f32 + beta * b.z as f32 + gamma * c.z as f32;
                    // println!("{} {} {} {} {} {} {}", a.z, b.z, c.z, alpha, beta, gamma, z);
                    // let z = z / 2.;
                    // let z = z as u8;
                    // let color = coloru8(z, z, z);
                    image.set(x as usize, y as usize, color);
                }
            }
        }
    }
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
}

fn main() -> std::io::Result<()> {
    let args = Args::parse();
    let model = {
        let mut model = wavefront_obj::Model::from_file(args.model.as_str()).unwrap();
        model.faces.reverse();
        model
    };

    println!(
        "Parsed {} vertices and {} faces",
        model.vertices.len(),
        model.faces.len()
    );

    let final_angle = if args.animate { 360 } else { 1 };
    for (idx, angle) in (0..final_angle).step_by(8).enumerate() {
        let mut image = Image::new(args.canvas_size, args.canvas_size);
        let mut depths = DepthBuffer::new(args.canvas_size, args.canvas_size);
        let canvas_size = args.canvas_size as f32;

        // let angle = angle + 180;
        let angle = angle as f32 * 2.0 * f32::consts::PI / 360.0;
        let s = angle.sin();
        let c = angle.cos();

        let light_dir = vec3(-0.2, 0.0, -1.).normalized();

        let colors = [
            ("white", WHITE),
            ("red", RED),
            ("green", GREEN),
            ("yellow", YELLOW),
            ("blue", BLUE),
            ("orange", ORANGE),
            ("pink", PINK),
            ("gold", GOLD),
        ];
        for (face_idx, face) in model.faces.iter().enumerate() {
            let mut screen_coords: [Vec3<i32>; 3] = [vec3(0, 0, 0); 3];
            let face_color = colors[face_idx % colors.len()];

            let mut world_coords: [Vec3<f32>; 3] = [vec3(0.0, 0.0, 0.0); 3];

            let d = 1.0;
            for j in 0..3 {
                let mut v = model.vertices[face[j]];
                let vx = c * v.x + s * v.z;
                let vz = -s * v.x + c * v.z;
                v.x = vx;
                v.z = vz;

                let vy = c * v.y + s * v.z;
                let vz = -s * v.y + c * v.z;
                v.y = vy;
                v.z = vz;

                // v.x += 0.8;
                // v.y -= 0.6;

                v.z += 2.0;

                assert!(v.z > 0.0);

                v.x = v.x * d / v.z;
                v.y = v.y * d / v.z;

                // println!("{:?} {}", v, face_color.0);

                // v.x = v.x * v.z * d;
                // v.y = v.y * v.z * d;

                screen_coords[j] = vec3(
                    ((v.x + 1.) * canvas_size / 2.0) as i32,
                    ((v.y + 1.) * canvas_size / 2.0) as i32,
                    ((v.z - 1.) * 250. / 2.0) as i32,
                );
                world_coords[j] = v;
            }

            let normal = ((world_coords[2] - world_coords[0])
                .cross(world_coords[1] - world_coords[0]))
            .normalized();

            let intensity = normal.dot(light_dir).abs();
            // let intensity = 0.99f31;

            let gray = (intensity * 255.0).clamp(0.0, 255.0) as u8;
            let triangle_color = coloru8(gray, gray, gray);
            // let triangle_color = WHITE;

            triangle(
                screen_coords[0],
                screen_coords[1],
                screen_coords[2],
                &mut image,
                &mut depths,
                triangle_color,
            );

            if args.wireframe {
                for i in 0..3 {
                    linevi32(
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

        // let depth_tga = tga::TgaFile::from_image(depths.to_image());
        // depth_tga.save_to_path(format!("raw-output/depth-{:02}.tga", idx).as_str())?;
    }

    Ok(())
}
