mod image;
mod tga;

use crate::image::*;

fn line(ax: usize, ay: usize, bx: usize, by: usize, image: &mut Image) {
    //
}

fn main() -> std::io::Result<()> {
    let mut image = Image::new(64u16, 64u16);

    let (ax, ay) = (7, 3);
    let (bx, by) = (12, 37);
    let (cx, cy) = (62, 53);

    image.set(ax, ay, WHITE);
    image.set(bx, by, WHITE);
    image.set(cx, cy, WHITE);

    let tga = tga::TgaFile::from_image(image);
    tga.save_to_path("output.tga")?;

    Ok(())
}
