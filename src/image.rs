use glam::U8Vec3;

pub type Color = U8Vec3;

pub const fn coloru8(r: u8, g: u8, b: u8) -> Color {
    U8Vec3::new(r, g, b)
}

pub const RED: Color = coloru8(255, 0, 0);

pub struct Image {
    buf: Vec<u8>,
    width: usize,
    height: usize,
}

impl Image {
    pub fn new(width: u16, height: u16) -> Self {
        let width = width as usize;
        let height = height as usize;
        let buf = vec![255u8; width * height * 4];
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

    pub fn buf_mut(&mut self) -> &mut Vec<u8> {
        &mut self.buf
    }

    #[inline]
    pub fn set(&mut self, x: usize, y: usize, color: Color) {
        let y = self.height - y - 1;
        let idx = (y * self.width + x) * 4;
        self.buf[idx] = color.x;
        self.buf[idx + 1] = color.y;
        self.buf[idx + 2] = color.z;
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

    pub fn buf_mut(&mut self) -> &mut Vec<f32> {
        &mut self.buf
    }

    pub fn buf(&self) -> &Vec<f32> {
        &self.buf
    }

    #[inline]
    pub fn width(&self) -> usize {
        self.width as _
    }

    #[inline]
    pub fn height(&self) -> usize {
        self.height as _
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
