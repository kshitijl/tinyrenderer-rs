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

    // We work our way up from A, which is (ax,ay).

    let mut y = ay;

    // First let's go up to the height of B.
    while y < by {
        assert!(by != ay);
        assert!(cy != ay);
        // Get the point on AB with this y.
        // When y = ay, x should be ax.
        // When y = by, x should be bx.
        let x_ab = ax as f32 + ((y - ay) as f32) / ((by - ay) as f32) * ((bx - ax) as f32);

        // Get the point on AC with this y.
        let x_ac = ax as f32 + ((y - ay) as f32) / ((cy - ay) as f32) * ((cx - ax) as f32);

        let lower_bound = f32::min(x_ab, x_ac).round() as i32;
        let upper_bound = f32::max(x_ab, x_ac).round() as i32;

        for x in lower_bound..=upper_bound {
            image.set(x as usize, y as usize, YELLOW);
        }

        y += 1;
    }

    assert!(y == by);

    // Then we'll go up to the height of C.
    while y <= cy {
        let x_bc = bx as f32 + ((y - by) as f32) / ((cy - by) as f32) * ((cx - bx) as f32);

        // Get the point on AC with this y.
        let x_ac = ax as f32 + ((y - ay) as f32) / ((cy - ay) as f32) * ((cx - ax) as f32);

        let lower_bound = f32::min(x_bc, x_ac).round() as i32;
        let upper_bound = f32::max(x_bc, x_ac).round() as i32;

        for x in lower_bound..=upper_bound {
            image.set(x as usize, y as usize, BLUE);
        }
        y += 1;
    }

    assert!(y == cy + 1);
}

fn triangle(ax: i32, ay: i32, bx: i32, by: i32, cx: i32, cy: i32, image: &mut Image, color: Color) {
    linei32(ax, ay, bx, by, image, color);
    linei32(bx, by, cx, cy, image, color);
    linei32(cx, cy, ax, ay, image, color);

    let mut arr = [(ax, ay), (bx, by), (cx, cy)];
    arr.sort_unstable_by_key(|p| p.1);
    fill_triangle(
        arr[0].0, arr[0].1, arr[1].0, arr[1].1, arr[2].0, arr[2].1, image, color,
    );
}

fn main() -> std::io::Result<()> {
    let s = 200;
    let mut image = Image::new(s, s);
    let s = image.width() as f32;

    triangle(7, 45, 35, 100, 45, 60, &mut image, RED);
    triangle(120, 35, 90, 5, 45, 110, &mut image, WHITE);
    triangle(115, 83, 80, 90, 85, 120, &mut image, GREEN);

    let tga = tga::TgaFile::from_image(image);
    let idx = 0;
    tga.save_to_path(format!("raw-output/frame-{:02}.tga", idx).as_str())?;

    Ok(())
}
