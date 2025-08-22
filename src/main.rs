mod tga;

use tga::TgaFile;
fn main() -> std::io::Result<()> {
    let w = 256u16;
    let h = 256u16;
    let mut rgb = vec![0u8; w as usize * h as usize * 3];

    for y in 0..h as usize {
        for x in 0..w as usize {
            let on = ((x / 32) ^ (y / 32)) & 1 == 0;
            let (r, g, b) = if on {
                if (x / 32 == 0) && (y / 32 == 0) {
                    (255, 0, 0)
                } else if y / 32 == 0 {
                    (0, 0, 255)
                } else {
                    (255, 255, 255)
                }
            } else {
                (0, 0, 0)
            };
            let i = (y * w as usize + x) * 3;
            rgb[i + 0] = r;
            rgb[i + 1] = g;
            rgb[i + 2] = b;
        }
    }

    let tga = TgaFile::from_rgb(w, h, &rgb);
    tga.save_to_path("output/checkers.tga")?;

    Ok(())
}
