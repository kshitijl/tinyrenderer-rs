use std::ops::{Add, Div, Index, IndexMut, Mul, Sub};

#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Vec2<T> {
    pub x: T,
    pub y: T,
}

pub fn vec2<T>(x: T, y: T) -> Vec2<T> {
    Vec2 { x, y }
}

impl<T: Add<Output = T>> Add for Vec2<T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}

impl<T: Sub<Output = T>> Sub for Vec2<T> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }
}

impl<T: Mul<Output = T> + Sub<Output = T>> Vec2<T> {
    pub fn cross(self, rhs: Self) -> T {
        self.x * rhs.y - self.y * rhs.x
    }
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Vec3<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}

pub fn vec3<T>(x: T, y: T, z: T) -> Vec3<T> {
    Vec3 { x, y, z }
}

impl<T> Vec3<T> {
    pub fn xy(self) -> Vec2<T> {
        Vec2 {
            x: self.x,
            y: self.y,
        }
    }

    pub fn to4w(self, w: T) -> Vec4<T> {
        Vec4 {
            x: self.x,
            y: self.y,
            z: self.z,
            w,
        }
    }
}

impl<T: Add<Output = T> + Mul<Output = T> + Sub<Output = T> + Copy> Vec3<T> {
    pub fn cross(self, rhs: Self) -> Self {
        Self {
            x: self.y * rhs.z - self.z * rhs.y,
            y: self.x * rhs.z - self.z * rhs.x,
            z: self.x * rhs.y - self.y * rhs.x,
        }
    }

    pub fn dot(self, rhs: Self) -> T {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }
}

impl Vec3<f32> {
    pub fn length(self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    pub fn normalized(self) -> Self {
        self / self.length()
    }

    pub fn to4(self) -> Vec4<f32> {
        self.to4w(1.0)
    }
}

impl<T: Add<Output = T>> Add for Vec3<T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl<T: Sub<Output = T>> Sub for Vec3<T> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl Div<f32> for Vec3<f32> {
    type Output = Self;

    fn div(self, rhs: f32) -> Self::Output {
        Self {
            x: self.x / rhs,
            y: self.y / rhs,
            z: self.z / rhs,
        }
    }
}

impl<T: Mul<Output = T> + Copy> Mul<T> for Vec3<T> {
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        Self {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
        }
    }
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Vec4<T> {
    pub x: T,
    pub y: T,
    pub z: T,
    pub w: T,
}

pub fn vec4<T>(x: T, y: T, z: T, w: T) -> Vec4<T> {
    Vec4 { x, y, z, w }
}

impl<T> Vec4<T> {
    pub fn xy(self) -> Vec2<T> {
        Vec2 {
            x: self.x,
            y: self.y,
        }
    }
    pub fn xyz(self) -> Vec3<T> {
        Vec3 {
            x: self.x,
            y: self.y,
            z: self.z,
        }
    }
}

impl<T: Add<Output = T> + Mul<Output = T> + Sub<Output = T> + Copy> Vec4<T> {
    pub fn dot(self, rhs: Self) -> T {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z + self.w * rhs.w
    }
}

impl Vec4<f32> {
    pub fn length(self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w).sqrt()
    }

    pub fn normalized(self) -> Self {
        self / self.length()
    }
}

impl<T: Add<Output = T>> Add for Vec4<T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
            w: self.w + rhs.w,
        }
    }
}

impl<T: Sub<Output = T>> Sub for Vec4<T> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
            w: self.w - rhs.w,
        }
    }
}

impl Div<f32> for Vec4<f32> {
    type Output = Self;

    fn div(self, rhs: f32) -> Self::Output {
        Self {
            x: self.x / rhs,
            y: self.y / rhs,
            z: self.z / rhs,
            w: self.w / rhs,
        }
    }
}

impl<T: Mul<Output = T> + Copy> Mul<T> for Vec4<T> {
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        Self {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
            w: self.w * rhs,
        }
    }
}

#[derive(Debug, PartialEq)]
pub struct Mat4<T> {
    rows: [[T; 4]; 4],
}

impl<T> Mat4<T> {
    pub fn new(rows: [[T; 4]; 4]) -> Self {
        Self { rows }
    }
}

impl Mat4<f32> {
    pub fn from_rotation_y(angle: f32) -> Self {
        let c = angle.cos();
        let s = angle.sin();

        Self::new([
            [c, 0., s, 0.],
            [0., 1., 0., 0.],
            [-s, 0., c, 0.],
            [0., 0., 0., 1.],
        ])
    }

    pub fn from_translation(t: Vec3<f32>) -> Self {
        Self::new([
            [1., 0., 0., t.x],
            [0., 1., 0., t.y],
            [0., 0., 1., t.z],
            [0., 0., 0., 1.],
        ])
    }

    pub fn from_shear(t: Vec3<f32>) -> Self {
        Self::new([
            [t.x, 0., 0., 0.],
            [0., t.y, 0., 0.],
            [0., 0., t.z, 0.],
            [0., 0., 0., 1.],
        ])
    }

    pub fn perspective(focal_length: f32) -> Self {
        Self::new([
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 0.],
        ])
    }
}

impl<T> Index<usize> for Mat4<T> {
    type Output = [T; 4];

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.rows[index]
    }
}

impl<T> IndexMut<usize> for Mat4<T> {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.rows[index]
    }
}

impl<T: Copy + Mul<Output = T> + Add<Output = T>> Mul<&Vec4<T>> for &Mat4<T> {
    type Output = Vec4<T>;

    #[inline]
    fn mul(self, vec: &Vec4<T>) -> Self::Output {
        vec4(
            self[0][0] * vec.x + self[0][1] * vec.y + self[0][2] * vec.z + self[0][3] * vec.w,
            self[1][0] * vec.x + self[1][1] * vec.y + self[1][2] * vec.z + self[1][3] * vec.w,
            self[2][0] * vec.x + self[2][1] * vec.y + self[2][2] * vec.z + self[2][3] * vec.w,
            self[3][0] * vec.x + self[3][1] * vec.y + self[3][2] * vec.z + self[3][3] * vec.w,
        )
    }
}

impl Mul<&Mat4<f32>> for &Mat4<f32> {
    type Output = Mat4<f32>;

    #[inline]
    fn mul(self, other: &Mat4<f32>) -> Self::Output {
        let mut result = [[0.0; 4]; 4];

        for i in 0..4 {
            for j in 0..4 {
                for k in 0..4 {
                    result[i][j] += self[i][k] * other[k][j];
                }
            }
        }

        Mat4::new(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let m1 = &Mat4::new([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 0.0],
            [0.0, 0.0, 0.0, 4.0],
        ]);

        let r = m1 * m1;

        let t = Mat4::new([
            [1., 0., 0., 0.],
            [0., 4., 0., 0.],
            [0., 0., 9., 0.],
            [0., 0., 0., 16.],
        ]);
        assert_eq!(r, t);
    }
}
