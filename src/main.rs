mod image;
mod linalg;
mod tga;
mod wavefront_obj;

use crate::image::*;
use crate::linalg::*;
use std::f32;
use std::ops::{Mul, Sub};

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
        let x_ab = ax + ((y - ay) * (bx - ax)) / (by - ay);

        let x_ac = ax + ((y - ay) * (cx - ax)) / (cy - ay);
        // Get the point on AC with this y.

        let lower_bound = i32::min(x_ab, x_ac);
        let upper_bound = i32::max(x_ab, x_ac);

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
fn triangle(ax: i32, ay: i32, bx: i32, by: i32, cx: i32, cy: i32, image: &mut Image, color: Color) {
    linei32(ax, ay, bx, by, image, color);
    linei32(bx, by, cx, cy, image, color);
    linei32(cx, cy, ax, ay, image, color);

    let mut arr = [(ax, ay), (bx, by), (cx, cy)];
    arr.sort_unstable_by_key(|p| p.1);
    fill_triangle(
        arr[0].0, arr[0].1, arr[1].0, arr[1].1, arr[2].0, arr[2].1, image, color,
    );
    // fill_triangle_pixeltest(ax, ay, bx, by, cx, cy, image, color);
}

fn main() -> std::io::Result<()> {
    let s = 200;
    let mut image = Image::new(s, s);
    let s = image.width() as f32;

    triangle(7, 45, 35, 100, 45, 60, &mut image, RED);
    triangle(120, 35, 90, 5, 45, 110, &mut image, WHITE);
    triangle(115, 83, 80, 90, 85, 120, &mut image, GREEN);
    triangle(115, 83, 80, 83, 85, 120, &mut image, RED);
    triangle(115, 83, 80, 83, 85, 83, &mut image, RED);
    triangle(115, 20, 115, 30, 140, 70, &mut image, RED);

    let tga = tga::TgaFile::from_image(image);
    let idx = 0;
    tga.save_to_path(format!("raw-output/frame-{:02}.tga", idx).as_str())?;

    Ok(())
}
