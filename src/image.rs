use glam::{U8Vec4, Vec3};

pub type Color = U8Vec4;

pub const fn coloru8(r: u8, g: u8, b: u8) -> Color {
    U8Vec4::new(r, g, b, 255)
}

pub fn colorvf(c: Vec3) -> Color {
    U8Vec4::new(c.x as u8, c.y as u8, c.z as u8, 255)
}

pub const RED: Color = coloru8(255, 0, 0);
pub const BLUE: Color = coloru8(0, 0, 255);
pub const GREY: Color = coloru8(127, 127, 127);
pub const GOLD: Color = coloru8(255, 155, 0);
pub const BLACK: Color = coloru8(0, 0, 0);

pub struct Image {
    buf: Vec<Color>,
    width: usize,
    height: usize,
}

pub trait WidthAndHeight {
    fn width(&self) -> usize;
    fn height(&self) -> usize;
}

pub trait ValidIndices: WidthAndHeight {
    #[inline]
    fn is_valid(&self, x: i32, y: i32) -> bool {
        x >= 0 && y >= 0 && x < self.width() as i32 && y < self.height() as i32
    }
}

impl ValidIndices for Image {}
impl ValidIndices for DepthBuffer {}

impl Image {
    pub fn new(width: u16, height: u16) -> Self {
        let width = width as usize;
        let height = height as usize;
        let buf = vec![RED; width * height];
        Self { width, height, buf }
    }

    pub fn clear(&mut self, c: Color) {
        self.buf.as_mut_slice().fill(c);
    }

    pub fn buf(&self) -> &Vec<Color> {
        &self.buf
    }

    #[inline]
    pub fn set(&mut self, x: usize, y: usize, color: Color) {
        let y = self.height - y - 1;
        let idx = y * self.width + x;
        self.buf[idx] = color;
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize) -> Color {
        let y = self.height - y - 1;
        let idx = y * self.width + x;
        self.buf[idx]
    }
}

impl WidthAndHeight for Image {
    #[inline]
    fn width(&self) -> usize {
        self.width as _
    }

    #[inline]
    fn height(&self) -> usize {
        self.height as _
    }
}

pub struct DepthBuffer {
    buf: Vec<f32>,
    width: usize,
    height: usize,
}

impl DepthBuffer {
    pub fn new(width: u16, height: u16) -> Self {
        let buf = vec![f32::MAX; width as usize * height as usize];
        Self {
            width: width as usize,
            height: height as usize,
            buf,
        }
    }

    pub fn clear(&mut self, v: f32) {
        self.buf.as_mut_slice().fill(v)
    }

    pub fn min_depth(&self) -> f32 {
        let min_depth = self
            .buf
            .iter()
            .min_by(|x, y| x.partial_cmp(y).unwrap())
            .unwrap_or(&f32::MIN);
        *min_depth
    }

    pub fn max_depth(&self) -> f32 {
        let max_depth = self
            .buf
            .iter()
            .filter(|&&x| x != f32::MAX)
            .max_by(|x, y| x.partial_cmp(y).unwrap())
            .unwrap_or(&f32::MAX);
        *max_depth
    }

    pub fn depth_to_u8(v: f32, min_depth: f32, max_depth: f32) -> u8 {
        let v = v.clamp(min_depth, max_depth);
        let v = (v - min_depth) / (max_depth - min_depth);

        (v * 255.) as u8
    }

    pub fn buf(&self) -> &Vec<f32> {
        &self.buf
    }

    #[inline]
    pub fn set(&mut self, x: usize, y: usize, val: f32) {
        let y = self.height - y - 1;
        let idx = y * self.width + x;
        self.buf[idx] = val;
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize) -> f32 {
        let y = self.height - y - 1;
        let idx = y * self.width + x;
        self.buf[idx]
    }
}

impl WidthAndHeight for DepthBuffer {
    #[inline]
    fn width(&self) -> usize {
        self.width as _
    }

    #[inline]
    fn height(&self) -> usize {
        self.height as _
    }
}
