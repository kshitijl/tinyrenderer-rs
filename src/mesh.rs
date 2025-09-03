mod wavefront_obj;

use glam::{Mat4, Vec3, Vec4, Vec4Swizzles, vec3, vec4};

#[derive(Debug, PartialEq, Clone)]
pub struct Face {
    vertices: [usize; 3],
    normals: [usize; 3],
}

#[derive(Clone)]
pub struct Mesh {
    vertices: Vec<Vec3>,
    faces: Vec<Face>,
    normals: Vec<Vec3>,
    bounding_box: (Vec3, Vec3),
    bounding_box_coords: [Vec4; 8],
}

impl Mesh {
    pub fn bounding_box(&self) -> (Vec3, Vec3) {
        self.bounding_box
    }

    pub fn bounding_box_coords(&self) -> &[Vec4; 8] {
        &self.bounding_box_coords
    }

    pub fn wall() -> Self {
        let vertices = vec![
            vec3(0., 1., 0.),
            vec3(1., 1., 0.),
            vec3(0., 0., 0.),
            vec3(1., 0., 0.),
        ];

        let normals = vec![vec3(0., 0., 1.)];

        let faces = vec![
            Face {
                vertices: [2, 1, 0],
                normals: [0, 0, 0],
            },
            Face {
                vertices: [1, 2, 3],
                normals: [0, 0, 0],
            },
        ];

        let mut answer = Self {
            vertices,
            faces,
            normals,

            bounding_box: (Vec3::ZERO, Vec3::ZERO),
            bounding_box_coords: [Vec4::ZERO; 8],
        };
        answer.normalize();

        answer
    }

    fn recompute_bb(&mut self) {
        let mut min = self.vertices[0];
        let mut max = self.vertices[0];

        for vertex in self.vertices.iter() {
            min.x = f32::min(min.x, vertex.x);
            min.y = f32::min(min.y, vertex.y);
            min.z = f32::min(min.z, vertex.z);

            max.x = f32::max(max.x, vertex.x);
            max.y = f32::max(max.y, vertex.y);
            max.z = f32::max(max.z, vertex.z);
        }

        self.bounding_box = (min, max);

        let mut idx = 0;
        for x in [min.x, max.x] {
            for y in [min.y, max.y] {
                for z in [min.z, max.z] {
                    self.bounding_box_coords[idx] = vec4(x, y, z, 1.0);
                    idx += 1;
                }
            }
        }
    }

    pub fn scale(&self) -> f32 {
        let bb = self.bounding_box();

        f32::max(bb.1.x - bb.0.x, f32::max(bb.1.y - bb.0.y, bb.1.z - bb.0.z))
    }

    pub fn normalize(&mut self) {
        self.recompute_bb();
        let bb = self.bounding_box();

        let m_trans = Mat4::from_translation(-(bb.1 + bb.0) / 2.);
        let scale = self.scale();
        let s = 2. / scale;
        let m_scale = Mat4::from_scale(vec3(s, s, s));

        let m_transform = m_scale * m_trans;

        for vertex in self.vertices.iter_mut() {
            *vertex = (m_transform * Vec4::from((*vertex, 1.0))).xyz();
        }

        self.recompute_bb();
    }
    pub fn num_faces(&self) -> usize {
        self.faces.len()
    }

    pub fn vertex(&self, face_idx: usize, vertex_idx: usize) -> Vec3 {
        self.vertices[self.faces[face_idx].vertices[vertex_idx]]
    }

    pub fn normal(&self, face_idx: usize, vertex_idx: usize) -> Vec3 {
        self.normals[self.faces[face_idx].normals[vertex_idx]]
    }

    pub fn from_file(filename: &str) -> std::io::Result<Self> {
        let wavefront_obj::Mesh {
            vertices,
            normals,
            faces,
        } = wavefront_obj::Mesh::from_file(filename)?;

        let mut answer = Self {
            vertices,
            normals,
            faces,
            bounding_box: (Vec3::ZERO, Vec3::ZERO),
            bounding_box_coords: [Vec4::ZERO; 8],
        };
        answer.recompute_bb();

        Ok(answer)
    }
}
