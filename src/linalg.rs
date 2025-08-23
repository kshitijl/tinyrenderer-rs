use std::ops::{Add, Div, Mul, Sub};

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
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

pub fn vec3(x: f32, y: f32, z: f32) -> Vec3 {
    Vec3 { x, y, z }
}

impl Add for Vec3 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl Div<f32> for Vec3 {
    type Output = Self;

    fn div(self, rhs: f32) -> Self::Output {
        Self {
            x: self.x / rhs,
            y: self.y / rhs,
            z: self.z / rhs,
        }
    }
}

impl Mul<f32> for Vec3 {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        Self {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
        }
    }
}
