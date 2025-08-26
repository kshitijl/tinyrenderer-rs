use crate::linalg::*;
use nom::{
    IResult, Parser,
    branch::alt,
    bytes::complete::tag,
    character::complete::{digit1, space0, space1, u32},
    combinator::{map, map_res},
    multi::{count, separated_list1},
    number::complete::float,
    sequence::{preceded, terminated},
};
use std::fs::File;
use std::io::{self, BufRead, BufReader};

pub struct Model {
    vertices: Vec<Vec3f>,
    faces: Vec<[usize; 3]>,
    pub normals: Vec<Vec3f>,
}

impl Model {
    pub fn from_file(filename: &str) -> io::Result<Self> {
        let file = File::open(filename)?;
        let reader = BufReader::new(file);

        let mut vertices = Vec::new();
        let mut faces = Vec::new();
        let mut normals = Vec::new();

        for line_result in reader.lines() {
            let line = line_result?;

            match parse_line(&line) {
                Ok((_remaining, data)) => {
                    if _remaining.len() != 0 {
                        println!("{}", _remaining);
                    }
                    match data {
                        ParsedLine::Vertex(v) => vertices.push(v),
                        ParsedLine::Triangle(f) => faces.push(f),
                        ParsedLine::Normal(n) => normals.push(n.normalized()),
                    }
                }

                Err(_e) => {
                    //
                }
            }
        }

        Ok(Self {
            vertices,
            normals,
            faces,
        })
    }
}

#[derive(Debug, PartialEq)]
enum ParsedLine {
    Vertex(Vec3<f32>),
    Triangle([usize; 3]),
    Normal(Vec3<f32>),
}

fn parse_line(input: &str) -> IResult<&str, ParsedLine> {
    preceded(space0, alt((parse_vertex, parse_face, parse_normal))).parse(input)
}

fn parse_face_triplet(input: &str) -> IResult<&str, u32> {
    terminated(u32, (tag("/"), digit1, tag("/"), digit1)).parse(input)
}

fn parse_tagged_vec3<'a, F, T>(kind: &str, f: F, input: &'a str) -> IResult<&'a str, T>
where
    F: Fn(Vec3f) -> T,
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
        |ids: Vec<u32>| -> Result<ParsedLine, _> {
            if ids.len() == 3 {
                let arr = [
                    ids[0] as usize - 1,
                    ids[1] as usize - 1,
                    ids[2] as usize - 1,
                ];
                Ok(ParsedLine::Triangle(arr))
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
}
