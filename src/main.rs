mod image;
mod tga;

use crate::image::{Image, color};

fn main() -> std::io::Result<()> {
    let mut image = Image::new(256u16, 256u16);
    let white = color(255, 255, 255);
    let red = color(255, 0, 0);
    let blue = color(0, 0, 255);
    let black = color(0, 0, 0);

    for y in 0..image.height() as usize {
        for x in 0..image.width() as usize {
            let on = ((x / 32) ^ (y / 32)) & 1 == 0;
            let color = if on {
                if (x / 32 == 0) && (y / 32 == 0) {
                    red
                } else if y / 32 == 0 {
                    blue
                } else {
                    white
                }
            } else {
                black
            };
            image.set(x, y, color);
        }
    }

    let tga = tga::TgaFile::from_image(image);
    tga.save_to_path("output.tga")?;

    Ok(())
}
