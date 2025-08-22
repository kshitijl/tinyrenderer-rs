use binrw::BinWrite;
use binrw::binrw;
use std::fs::File;
use std::io;

#[binrw]
#[brw(little)]
#[derive(Debug)]
struct TgaHeader {
    id_length: u8,
    color_map_type: u8,
    image_type: u8,

    color_map_origin: u16,
    color_map_length: u16,
    color_map_depth: u8,

    x_origin: u16,
    y_origin: u16,
    width: u16,
    height: u16,

    pixel_depth: u8,

    image_descriptor: u8,
}

impl TgaHeader {
    fn new(width: u16, height: u16) -> Self {
        let pixel_depth = 24;
        let alpha_bits = 0;
        let image_descriptor = alpha_bits & 0x0F | 1 << 5; // top-left origin

        Self {
            id_length: 0,
            color_map_type: 0,
            image_type: 2, // uncompressed true-color
            color_map_origin: 0,
            color_map_length: 0,
            color_map_depth: 0,
            x_origin: 0,
            y_origin: 0,
            width,
            height,
            pixel_depth,
            image_descriptor,
        }
    }

    fn bytes_per_pixel(&self) -> usize {
        (self.pixel_depth as usize + 7) / 8
    }
}

#[binrw]
#[brw(little)]
#[derive(Debug, Clone)]
struct TgaFooter {
    extension_offset: u32,
    developer_dir_offset: u32,

    #[brw(pad_size_to = 16)]
    signature16: [u8; 16],
    dot: u8,
    nul: u8,
}

impl Default for TgaFooter {
    fn default() -> Self {
        let mut sig = [0u8; 16];
        sig[..16].copy_from_slice(b"TRUEVISION-XFILE");
        Self {
            extension_offset: 0,
            developer_dir_offset: 0,
            signature16: sig,
            dot: b'.',
            nul: 0,
        }
    }
}

#[binrw]
#[brw(little)]
#[derive(Debug)]
pub struct TgaFile {
    header: TgaHeader,

    #[br(count = header.id_length as usize)]
    image_id: Vec<u8>,

    #[br(count = (header.width as usize) * (header.height as usize) * header.bytes_per_pixel())]
    pixel_data: Vec<u8>,

    footer: TgaFooter,
}

impl TgaFile {
    pub fn from_rgb(width: u16, height: u16, rgb: &[u8]) -> Self {
        assert_eq!(rgb.len(), (width as usize) * (height as usize) * 3);

        let header = TgaHeader::new(width, height);

        let src_stride = (width as usize) * 3;
        let dst_bpp = header.bytes_per_pixel();
        let mut pixel_data = vec![0u8; (width as usize) * (height as usize) * dst_bpp];

        for y in 0..(height as usize) {
            let src_y = y;
            let dst_y = y;

            let src_row = &rgb[src_y * src_stride..src_y * src_stride + src_stride];
            let dst_row = &mut pixel_data
                [dst_y * (width as usize) * dst_bpp..(dst_y + 1) * (width as usize) * dst_bpp];

            for x in 0..(width as usize) {
                let r = src_row[x * 3 + 0];
                let g = src_row[x * 3 + 1];
                let b = src_row[x * 3 + 2];

                // TGA wants B, G, R, (A)
                dst_row[x * dst_bpp + 0] = b;
                dst_row[x * dst_bpp + 1] = g;
                dst_row[x * dst_bpp + 2] = r;
            }
        }
        Self {
            header,
            image_id: Vec::new(),
            pixel_data,
            footer: TgaFooter::default(),
        }
    }

    pub fn save_to_path(&self, path: &str) -> io::Result<()> {
        let mut f = File::create(path)?;
        self.write(&mut f)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        Ok(())
    }
}
