use crate::linalg::*;
use nom::{
    IResult, Parser,
    branch::alt,
    bytes::complete::tag,
    character::complete::{digit1, space0, space1, u32},
    combinator::{map, map_res},
    multi::separated_list1,
    number::complete::float,
    sequence::{preceded, terminated},
};
use std::fs::File;
use std::io::{self, BufRead, BufReader};

pub struct Model {
    pub vertices: Vec<Vec3<f32>>,
    pub faces: Vec<[usize; 3]>,
}

impl Model {
    pub fn from_file(filename: &str) -> io::Result<Self> {
        let file = File::open(filename)?;
        let reader = BufReader::new(file);

        let mut vertices = Vec::new();
        let mut faces = Vec::new();

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
                    }
                }

                Err(_e) => {
                    //
                }
            }
        }

        Ok(Self { vertices, faces })
    }
}

#[derive(Debug, PartialEq)]
enum ParsedLine {
    Vertex(Vec3<f32>),
    Triangle([usize; 3]),
}

fn parse_face_triplet(input: &str) -> IResult<&str, u32> {
    terminated(u32, (tag("/"), digit1, tag("/"), digit1)).parse(input)
}

fn parse_vertex(input: &str) -> IResult<&str, ParsedLine> {
    map(
        preceded(
            (tag("v"), space1),
            (float, preceded(space1, float), preceded(space1, float)),
        ),
        |(x, y, z)| ParsedLine::Vertex(vec3(x, y, z)),
    )
    .parse(input)
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

fn parse_line(input: &str) -> IResult<&str, ParsedLine> {
    preceded(space0, alt((parse_vertex, parse_face))).parse(input)
}
