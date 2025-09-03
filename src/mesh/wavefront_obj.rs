use super::Face;
use glam::{Vec3, vec3};
use nom::{
    IResult, Parser,
    branch::alt,
    bytes::complete::tag,
    character::complete::{digit0, space0, space1, u32},
    combinator::map_res,
    multi::separated_list1,
    number::complete::float,
    sequence::{preceded, terminated},
};
use std::fs::File;
use std::io::{self, BufRead, BufReader};

pub struct Mesh {
    pub vertices: Vec<Vec3>,
    pub faces: Vec<Face>,
    pub normals: Vec<Vec3>,
}

#[derive(Debug, PartialEq)]
enum ParsedLine {
    Vertex(Vec3),
    Triangle(Face),
    Normal(Vec3),
}
impl Mesh {
    pub fn from_file(filename: &str) -> io::Result<Mesh> {
        let file = File::open(filename)?;
        let reader = BufReader::new(file);

        let mut vertices = Vec::new();
        let mut faces = Vec::new();
        let mut normals = Vec::new();

        for line_result in reader.lines() {
            let line = line_result?;

            match parse_line(&line) {
                Ok((remaining, data)) => {
                    if !remaining.is_empty() {
                        println!("{}", remaining);
                    }
                    match data {
                        ParsedLine::Vertex(v) => vertices.push(v),
                        ParsedLine::Triangle(f) => faces.push(f),
                        ParsedLine::Normal(n) => normals.push(n.normalize()),
                    }
                }

                Err(_e) => {
                    //
                }
            }
        }

        Ok(Mesh {
            vertices,
            faces,
            normals,
        })
    }

    #[allow(dead_code)]
    pub fn num_vertices(&self) -> usize {
        self.vertices.len()
    }

    #[allow(dead_code)]
    pub fn num_normals(&self) -> usize {
        self.normals.len()
    }

    #[allow(dead_code)]
    pub fn num_faces(&self) -> usize {
        self.faces.len()
    }
}

fn parse_line(input: &str) -> IResult<&str, ParsedLine> {
    preceded(space0, alt((parse_vertex, parse_face, parse_normal))).parse(input)
}

fn parse_face_triplet(input: &str) -> IResult<&str, (u32, &str, u32)> {
    (terminated(u32, tag("/")), terminated(digit0, tag("/")), u32).parse(input)
}

fn parse_tagged_vec3<'a, F, T>(kind: &str, f: F, input: &'a str) -> IResult<&'a str, T>
where
    F: Fn(Vec3) -> T,
{
    let (input, numbers) = preceded(
        tag(kind),
        (
            preceded(space1, float),
            preceded(space1, float),
            preceded(space1, float),
        ),
    )
    .parse(input)?;

    let (x, y, z) = numbers;

    Ok((input, f(vec3(x, y, z))))
}

fn parse_normal(input: &str) -> IResult<&str, ParsedLine> {
    parse_tagged_vec3("vn", ParsedLine::Normal, input)
}

fn parse_vertex(input: &str) -> IResult<&str, ParsedLine> {
    parse_tagged_vec3("v", ParsedLine::Vertex, input)
}

fn parse_face(input: &str) -> IResult<&str, ParsedLine> {
    map_res(
        preceded(
            (tag("f"), space1),
            separated_list1(space1, parse_face_triplet),
        ),
        |ids: Vec<(u32, &str, u32)>| -> Result<ParsedLine, _> {
            if ids.len() == 3 {
                let vertices = [
                    ids[0].0 as usize - 1,
                    ids[1].0 as usize - 1,
                    ids[2].0 as usize - 1,
                ];

                let normals = [
                    ids[0].2 as usize - 1,
                    ids[1].2 as usize - 1,
                    ids[2].2 as usize - 1,
                ];

                let face = Face { vertices, normals };

                Ok(ParsedLine::Triangle(face))
            } else {
                Err("Face must have exactly 3 vertices")
            }
        },
    )
    .parse(input)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_parses_vertex() {
        let p = parse_vertex("v -0.000581696 -0.734665 -0.623267")
            .unwrap()
            .1;

        assert_eq!(
            p,
            ParsedLine::Vertex(vec3(-0.000581696, -0.734665, -0.623267))
        );
    }

    #[test]
    fn it_parses_vertex_normal() {
        let p = parse_normal("vn -0.000581696 -0.734665 -0.623267")
            .unwrap()
            .1;

        assert_eq!(
            p,
            ParsedLine::Normal(vec3(-0.000581696, -0.734665, -0.623267))
        );
    }

    #[test]
    fn it_parses_face() {
        let p = parse_face("f 83/64/83 96/74/96 95/73/95").unwrap().1;

        assert_eq!(
            p,
            ParsedLine::Triangle(Face {
                vertices: [82, 95, 94],
                normals: [82, 95, 94]
            })
        );
    }

    #[test]
    fn it_parses_face_without_uv() {
        let p = parse_face("f 5//1 3//1 1//1").unwrap().1;

        assert_eq!(
            p,
            ParsedLine::Triangle(Face {
                vertices: [4, 2, 0],
                normals: [0, 0, 0]
            })
        );
    }

    #[test]
    fn it_parses_files() {
        let model = Mesh::from_file("./assets/head.obj").unwrap();

        assert_eq!(model.num_faces(), 2492);
        assert_eq!(model.num_vertices(), 1258);
        assert_eq!(model.num_normals(), 1258);

        let model = Mesh::from_file("./assets/cube.obj").unwrap();

        assert_eq!(model.num_faces(), 8);
        assert_eq!(model.num_vertices(), 8);
        assert_eq!(model.num_normals(), 0);
    }
}
