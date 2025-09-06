use ena::unify::{InPlaceUnificationTable, UnifyKey};
use rand::{self, seq::SliceRandom};
use smallvec::SmallVec;
use std::collections::HashSet;

#[derive(Copy, Debug, Clone, PartialEq)]
pub enum GridElem {
    Wall,
    Empty,
    Exhibit,
}
pub struct FloorPlan {
    width: u32,
    height: u32,
    grid: Vec<GridElem>,
}

#[derive(Clone, Debug, PartialEq, Copy, Eq, Hash, Default, Ord, PartialOrd)]
struct GridIdx(u32);

impl UnifyKey for GridIdx {
    type Value = ();

    fn index(&self) -> u32 {
        self.0
    }

    fn from_index(u: u32) -> GridIdx {
        GridIdx(u)
    }

    fn tag() -> &'static str {
        "GridIdx"
    }
}

impl FloorPlan {
    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    fn to_xy(&self, g: GridIdx) -> (u32, u32) {
        let x = g.0 % self.width();
        let y = g.0 / self.width();
        (x, y)
    }

    fn from_xy(&self, x: u32, y: u32) -> GridIdx {
        GridIdx(y * self.width() + x)
    }

    pub fn valid_neighbors(&self, g: GridIdx) -> SmallVec<[GridIdx; 4]> {
        let mut answer = SmallVec::new();
        let (x, y) = self.to_xy(g);

        for (dx, dy) in [(1, 0), (-1i32, 0), (0, 1), (0, -1i32)] {
            let (neighbor_x, neighbor_y) = (x as i32 + dx, y as i32 + dy);
            if self.is_valid(neighbor_x, neighbor_y) {
                answer.push(self.from_xy(neighbor_x as u32, neighbor_y as u32));
            }
        }

        answer
    }

    fn at(&self, g: GridIdx) -> GridElem {
        self.grid[g.0 as usize]
    }

    fn is_valid(&self, x: i32, y: i32) -> bool {
        x >= 0 && y >= 0 && x < self.width as i32 && y < self.height as i32
    }

    fn new_all_walls(width: u32, height: u32) -> Self {
        let grid = vec![GridElem::Wall; (width * height) as usize];
        Self {
            width,
            height,
            grid,
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, Hash, PartialEq)]
struct Wall {
    a: GridIdx,
    b: GridIdx,
}

fn generate_maze(width: u32, height: u32) -> String {
    // let mut f = Self::new_all_walls(width, height);

    let f = FloorPlan {
        width,
        height,
        grid: Vec::new(),
    };

    let mut u: InPlaceUnificationTable<GridIdx> = InPlaceUnificationTable::new();

    for x in 0..width {
        for y in 0..height {
            u.new_key(());
        }
    }

    let mut all_walls: Vec<Wall> = Vec::new();

    for x in 0..width {
        for y in 0..height {
            let this_cell = f.from_xy(x, y);
            if x < width - 1 {
                let right_neighbor = f.from_xy(x + 1, y);
                all_walls.push(Wall {
                    a: this_cell,
                    b: right_neighbor,
                });
            }
            if y < height - 1 {
                let bottom_neighbor = f.from_xy(x, y + 1);
                all_walls.push(Wall {
                    a: this_cell,

                    b: bottom_neighbor,
                });
            }
        }
    }

    let mut rng = rand::rng();
    all_walls.shuffle(&mut rng);

    let mut deleted_walls: HashSet<Wall> = HashSet::new();
    for wall in all_walls.iter() {
        let Wall { a, b } = *wall;
        let set_a = u.find(a);
        let set_b = u.find(b);

        if set_a != set_b {
            u.union(a, b);
            deleted_walls.insert(*wall);
        }
    }

    let mut answer: Vec<String> = Vec::new();

    for y in 0..height {
        let mut toprow: Vec<char> = Vec::new();
        let mut botrow: Vec<char> = Vec::new();

        for x in 0..width {
            let me = f.from_xy(x, y);
            let right_neighbor = f.from_xy(x + 1, y);
            let bottom_neighbor = f.from_xy(x, y + 1);

            let tl = '.';
            let mut tr = '.';
            let mut bl = '.';
            let mut br = '.';

            if deleted_walls.contains(&Wall {
                a: me,
                b: bottom_neighbor,
            }) {
                print!(" ");
            } else {
                bl = 'w';
                br = 'w';
                print!("_");
            }
            if deleted_walls.contains(&Wall {
                a: me,
                b: right_neighbor,
            }) {
                print!(" ");
            } else {
                tr = 'w';
                br = 'w';
                print!("|");
            }

            toprow.push(tl);
            toprow.push(tr);
            botrow.push(bl);
            botrow.push(br);
        }

        let toprow_s: String = toprow.iter().collect();
        let botrow_s: String = botrow.iter().collect();

        answer.push(toprow_s);
        answer.push(botrow_s);

        print!("\n");
    }

    for row in answer {
        println!("{}", row);
    }

    "".to_string()
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_generates() {
        generate_maze(16, 16);

        assert_eq!(1, 0);
    }
}
