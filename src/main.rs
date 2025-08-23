mod image;
mod linalg;
mod tga;
mod wavefront_obj;

use std::f32;

use crate::image::*;
use crate::linalg::*;

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
    let mut x = ax;
    let mut y = ay;
    let mut ierror = 0; // defined as error * 2 * (bx - ax)
    let dy = if by > ay { 1 } else { -1 };
    while x <= bx {
        let (xx, yy) = if !steep { (x, y) } else { (y, x) };

        assert!(xx >= 0);
        assert!(yy >= 0);
        image.set(xx as usize, yy as usize, color);

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

fn main() -> std::io::Result<()> {
    let s = 800u16;

    let model = wavefront_obj::Model::from_file("assets/diablo3_pose.obj").unwrap();
    println!(
        "Parsed model with {} vertices and {} triangles",
        model.vertices.len(),
        model.faces.len()
    );

    let one = vec3(1.0, 1.0, 1.0);

    for (idx, angle) in (0..360).step_by(20).enumerate() {
        let mut image = Image::new(s, s);
        let s = image.width() as f32;

        let angle = angle as f32 * 2.0 * f32::consts::PI / 360.0;
        let cos_angle = angle.cos();
        let sin_angle = angle.sin();

        for face in model.faces.iter() {
            let mut a = model.vertices[face[0]];
            let mut b = model.vertices[face[1]];
            let mut c = model.vertices[face[2]];

            a.x = a.x * cos_angle + a.z * sin_angle;
            b.x = b.x * cos_angle + b.z * sin_angle;
            c.x = c.x * cos_angle + c.z * sin_angle;

            let a = (one + a) * s / 2.0001;
            let b = (one + b) * s / 2.0001;
            let c = (one + c) * s / 2.0001;

            linef32(a.x, a.y, b.x, b.y, &mut image, RED);
            linef32(b.x, b.y, c.x, c.y, &mut image, RED);
            linef32(c.x, c.y, a.x, a.y, &mut image, RED);
        }
        let tga = tga::TgaFile::from_image(image);
        tga.save_to_path(format!("output-{:02}.tga", idx).as_str())?;
    }

    Ok(())
}
