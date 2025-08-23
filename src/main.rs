mod image;
mod linalg;
mod tga;
mod wavefront_obj;

use crate::image::*;
use crate::linalg::*;

fn line(ax: i32, ay: i32, bx: i32, by: i32, image: &mut Image, color: Color) {
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
    // let mut y = ay;
    // let mut ierror = 0; // defined as error * 2 * (bx - ax)
    // let dy = if by > ay { 1 } else { -1 };
    while x <= bx {
        let t = (x - ax) as f32 / (bx - ax) as f32;

        let y = (ay as f32 + (by - ay) as f32 * t).round() as i32;
        let (xx, yy) = if !steep { (x, y) } else { (y, x) };

        assert!(xx >= 0);
        assert!(yy >= 0);
        image.set(xx as usize, yy as usize, color);

        // ierror += (by - ay) * 2;

        // let should_incr = (ierror > (bx - ax)) as i32;
        // y += dy * should_incr;
        // ierror -= 2 * (bx - ax) * should_incr;
        x += 1;
    }
}

fn main() -> std::io::Result<()> {
    let s = 2048u16;
    let mut image = Image::new(s, s);

    let model = wavefront_obj::Model::from_file("assets/diablo3_pose.obj").unwrap();
    println!(
        "Parsed model with {} vertices and {} triangles",
        model.vertices.len(),
        model.faces.len()
    );

    println!("{:?}", model.vertices[0]);
    println!("{:?}", model.faces[0]);

    let s = image.width() as f32;
    let one = vec3(1.0, 1.0, 1.0);
    for face in model.faces.iter() {
        let a = (one + model.vertices[face[0]]) / 2.0001;
        let b = (one + model.vertices[face[1]]) / 2.0001;
        let c = (one + model.vertices[face[2]]) / 2.0001;

        line(
            (a.x * s) as i32,
            (a.y * s) as i32,
            (b.x * s) as i32,
            (b.y * s) as i32,
            &mut image,
            RED,
        );
        line(
            (b.x * s) as i32,
            (b.y * s) as i32,
            (c.x * s) as i32,
            (c.y * s) as i32,
            &mut image,
            RED,
        );
        line(
            (c.x * s) as i32,
            (c.y * s) as i32,
            (a.x * s) as i32,
            (a.y * s) as i32,
            &mut image,
            RED,
        );
    }

    for vertex in model.vertices.iter() {
        let v = (one + *vertex) / 2.0001;
        image.set((v.x * s) as usize, (v.y * s) as usize, WHITE);
    }
    let tga = tga::TgaFile::from_image(image);
    tga.save_to_path("output.tga")?;

    Ok(())
}
