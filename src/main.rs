mod image;
mod tga;

use crate::image::*;

fn line(ax: usize, ay: usize, bx: usize, by: usize, image: &mut Image, color: Color) {
    let (ax, ay, bx, by) = (ax as i32, ay as i32, bx as i32, by as i32);
    let (ax, bx, ay, by) = if ax <= bx {
        (ax, bx, ay, by)
    } else {
        (bx, ax, by, ay)
    };

    let steep = (by - ay).abs() > (bx - ax).abs();
    let (ax, bx, ay, by) = if !steep {
        (ax, bx, ay, by)
    } else {
        (ay, by, ax, bx)
    };

    let mut x = ax;
    let mut y = ay;
    let mut ierror = 0; // defined as error * 2 * (bx - ax)
    let dy = if by > ay { 1 } else { -1 };
    while x <= bx {
        let (xx, yy) = if !steep { (x, y) } else { (y, x) };
        image.set(xx as usize, yy as usize, color);

        ierror += (by - ay) * 2;

        let should_incr = (ierror > (bx - ax)) as i32;
        y += dy * should_incr;
        ierror -= 2 * (bx - ax) * should_incr;
        x += 1;
    }
}

fn main() -> std::io::Result<()> {
    let mut image = Image::new(64u16, 64u16);

    let (ax, ay) = (7, 3);
    let (bx, by) = (12, 37);
    let (cx, cy) = (62, 53);

    line(ax, ay, bx, by, &mut image, BLUE);
    line(cx, cy, bx, by, &mut image, GREEN);
    line(cx, cy, ax, ay, &mut image, YELLOW);
    line(ax, ay, cx, cy, &mut image, RED);

    image.set(ax, ay, WHITE);
    image.set(bx, by, WHITE);
    image.set(cx, cy, WHITE);

    let tga = tga::TgaFile::from_image(image);
    tga.save_to_path("output.tga")?;

    Ok(())
}
