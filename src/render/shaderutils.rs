use glam::{Vec3, vec3};

#[allow(dead_code)]
pub fn step(edge: f32, t: f32) -> f32 {
    if t < edge { 0. } else { 1. }
}

pub fn smoothstep(edge0: f32, edge1: f32, t: f32) -> f32 {
    let t = ((t - edge0) / (edge1 - edge0)).clamp(0., 1.);
    t * t * (3. - 2. * t)
}

pub fn hue2rgb(p: f32, q: f32, t: f32) -> f32 {
    let t = if t < 0. { t + 1. } else { t };
    let t = if t > 1. { t - 1. } else { t };
    if t < 1. / 6. {
        return p + (q - p) * 6. * t;
    }
    if t < 1. / 2. {
        return q;
    }
    if t < 2. / 3. {
        return p + (q - p) * (2. / 3. - t) * 6.;
    }

    p
}

pub fn hsl2rgb(h: f32, s: f32, l: f32) -> Vec3 {
    if s < 1e-6 {
        // achromatic
        vec3(l, l, l)
    } else {
        let q = if l < 0.5 { l * (s + 1.) } else { l + s - l * s };
        let p = 2. * l - q;
        let r = hue2rgb(p, q, h + 1. / 3.);
        let g = hue2rgb(p, q, h);
        let b = hue2rgb(p, q, h - 1. / 3.);
        vec3(r, g, b)
    }
}
