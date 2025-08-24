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
pub struct Vec3<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}

pub fn vec3<T>(x: T, y: T, z: T) -> Vec3<T> {
    Vec3 { x, y, z }
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
