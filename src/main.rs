mod image;
mod tga;

use crate::image::*;

fn line(ax: usize, ay: usize, bx: usize, by: usize, image: &mut Image, color: Color) {
    let mut x = ax;
    while x <= bx {
        let t = (x as i32 - ax as i32) as f32 / (bx as i32 - ax as i32) as f32;
        let y = (ay as f32 + (by as i32 - ay as i32) as f32 * t).round();

        image.set(x as _, y as _, color);

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
