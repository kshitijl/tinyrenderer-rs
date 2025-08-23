#[derive(Copy, Clone)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

pub const fn color(r: u8, g: u8, b: u8) -> Color {
    Color { r, g, b }
}

pub const WHITE: Color = color(255, 255, 255);
pub const RED: Color = color(255, 0, 0);
pub const GREEN: Color = color(0, 255, 0);
pub const YELLOW: Color = color(255, 255, 0);
pub const BLUE: Color = color(0, 0, 255);
pub const BLACK: Color = color(0, 0, 0);

pub struct Image {
    buf: Vec<u8>,
    width: u16,
    height: u16,
}

impl Image {
    pub fn new(width: u16, height: u16) -> Self {
        let buf = vec![0u8; width as usize * height as usize * 3];
        Self { width, height, buf }
    }

    pub fn width(&self) -> usize {
        self.width as _
    }

    pub fn height(&self) -> usize {
        self.height as _
    }

    pub fn buf(&self) -> &Vec<u8> {
        &self.buf
    }

    pub fn set(&mut self, x: usize, y: usize, color: Color) {
        let idx = (y * self.width as usize + x) * 3;
        self.buf[idx + 0] = color.r;
        self.buf[idx + 1] = color.g;
        self.buf[idx + 2] = color.b;
    }
}
