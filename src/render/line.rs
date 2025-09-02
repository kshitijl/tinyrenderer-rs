use crate::image::Color;
use crate::image::Image;
use crate::render::RenderingResult;
use glam::Vec2;

fn linei32(ax: i32, ay: i32, bx: i32, by: i32, image: &mut Image, color: Color) -> RenderingResult {
    let mut answer = RenderingResult::new();

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
            answer.num_pixels_drawn += 1;
            image.set(xx as usize, yy as usize, color);
        }

        ierror += (by - ay).abs() * 2;
        let should_incr = (ierror > (bx - ax)) as i32;
        y += dy * should_incr;
        ierror -= 2 * (bx - ax) * should_incr;
        x += 1;
    }

    answer
}

fn linef32(ax: f32, ay: f32, bx: f32, by: f32, image: &mut Image, color: Color) -> RenderingResult {
    linei32(ax as i32, ay as i32, bx as i32, by as i32, image, color)
}

pub fn linevf32(a: Vec2, b: Vec2, image: &mut Image, color: Color) -> RenderingResult {
    linef32(a.x, a.y, b.x, b.y, image, color)
}
