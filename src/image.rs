use glam::U8Vec3;

pub type Color = U8Vec3;

pub const fn coloru8(r: u8, g: u8, b: u8) -> Color {
    U8Vec3::new(r, g, b)
}

pub const WHITE: Color = coloru8(255, 255, 255);
pub const RED: Color = coloru8(255, 0, 0);
pub const GREEN: Color = coloru8(0, 255, 0);
pub const YELLOW: Color = coloru8(255, 255, 0);
pub const BLUE: Color = coloru8(0, 0, 255);
pub const BLACK: Color = coloru8(0, 0, 0);
pub const ORANGE: Color = coloru8(0xff, 0x45, 0x00);
pub const PINK: Color = coloru8(0xff, 0xc0, 0xcb);
pub const GOLD: Color = coloru8(0xff, 0xd7, 0x00);

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

    #[inline]
    pub fn width(&self) -> usize {
        self.width as _
    }

    #[inline]
    pub fn height(&self) -> usize {
        self.height as _
    }

    pub fn buf(&self) -> &Vec<u8> {
        &self.buf
    }

    #[inline]
    pub fn set(&mut self, x: usize, y: usize, color: Color) {
        let idx = (y * self.width as usize + x) * 3;
        self.buf[idx + 0] = color.x;
        self.buf[idx + 1] = color.y;
        self.buf[idx + 2] = color.z;
    }
}

pub struct DepthBuffer {
    buf: Vec<f32>,
    width: u16,
    height: u16,
}

impl DepthBuffer {
    pub fn new(width: u16, height: u16) -> Self {
        let buf = vec![f32::MAX; width as usize * height as usize];
        Self { width, height, buf }
    }

    pub fn to_image(&self) -> Image {
        let mut image = Image::new(self.width, self.height);

        let min_depth = self
            .buf
            .iter()
            .min_by(|x, y| x.partial_cmp(&y).unwrap())
            .unwrap();

        let max_depth = self
            .buf
            .iter()
            .filter(|&&x| x != f32::MAX)
            .max_by(|x, y| x.partial_cmp(&y).unwrap())
            .unwrap();

        for x in 0..self.width as usize {
            for y in 0..self.height as usize {
                let v = self.buf[y * self.width as usize + x];

                let v = v.clamp(*min_depth, *max_depth);
                let v = (v - *min_depth) / (max_depth - min_depth);
                let v = (v * 255.) as u8;
                let v = 255 - v;
                let color = coloru8(v, v, v);
                image.set(x, y, color);
            }
        }

        image
    }

    #[inline]
    pub fn width(&self) -> usize {
        self.width as _
    }

    #[inline]
    pub fn height(&self) -> usize {
        self.height as _
    }

    // pub fn buf(&self) -> &Vec<f32> {
    //     &self.buf
    // }

    #[inline]
    pub fn set(&mut self, x: usize, y: usize, val: f32) {
        let idx = y * self.width as usize + x;
        self.buf[idx] = val;
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize) -> f32 {
        let idx = y * self.width as usize + x;
        self.buf[idx]
    }
}
