use ena::unify::{InPlaceUnificationTable, UnifyKey};
use rand::{self, Rng, seq::SliceRandom};
use smallvec::SmallVec;
use std::collections::{HashMap, HashSet};

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

    fn set(&mut self, g: GridIdx, v: GridElem) {
        self.grid[g.0 as usize] = v;
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

    fn all_cells(&self) -> Vec<GridIdx> {
        let mut answer = Vec::new();
        for y in 0..self.height {
            for x in 0..self.width {
                answer.push(self.from_xy(x, y));
            }
        }
        answer
    }

    fn room_would_fit(&self, x1: i32, y1: i32, x2: i32, y2: i32) -> bool {
        // The +1 and -1 here are to stop stamping out the boundary wall.
        self.is_valid(x1 - 1, y1 - 1) && self.is_valid(x2 + 1, y2 + 1)
    }

    fn stamp_room(&mut self, x1: i32, y1: i32, x2: i32, y2: i32) {
        for x in x1..=x2 {
            for y in y1..=y2 {
                self.set(self.from_xy(x as u32, y as u32), GridElem::Empty);
            }
        }
    }
    fn print(&self) {
        for y in 0..self.height {
            for x in 0..self.width {
                let c = match self.at(self.from_xy(x, y)) {
                    GridElem::Wall => "w",
                    GridElem::Empty => ".",
                    GridElem::Exhibit => "x",
                };
                print!("{}", c);
            }
            print!("\n");
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, Hash, PartialEq)]
struct Wall {
    a: GridIdx,
    b: GridIdx,
}

#[derive(Copy, Debug, PartialEq, Clone, Eq, Hash)]
struct UFKey(u32);

impl UnifyKey for UFKey {
    type Value = ();

    fn index(&self) -> u32 {
        self.0
    }

    fn from_index(u: u32) -> UFKey {
        UFKey(u)
    }

    fn tag() -> &'static str {
        "GridIdx"
    }
}
struct UFWithOwnKey {
    unify: InPlaceUnificationTable<UFKey>,
    ok2uk: HashMap<GridIdx, UFKey>,
    uk2ok: HashMap<UFKey, GridIdx>,
}

impl UFWithOwnKey {
    fn new() -> Self {
        Self {
            unify: InPlaceUnificationTable::new(),
            ok2uk: HashMap::new(),
            uk2ok: HashMap::new(),
        }
    }

    fn find(&mut self, g: GridIdx) -> Option<GridIdx> {
        if let Some(key) = self.ok2uk.get(&g) {
            let root = self.unify.find(*key);
            self.uk2ok.get(&root).copied()
        } else {
            None
        }
    }

    fn insert(&mut self, g: GridIdx) {
        let key = self.unify.new_key(());
        self.ok2uk.insert(g, key);
        self.uk2ok.insert(key, g);
    }

    fn union(&mut self, k1: GridIdx, k2: GridIdx) {
        let uk1 = self.ok2uk.get(&k1).unwrap();
        let uk2 = self.ok2uk.get(&k2).unwrap();
        self.unify.union(*uk1, *uk2)
    }
}

fn generate_maze(
    width: u32,
    height: u32,
    num_rooms: u32,
    room_size: i32,
    room_size_variance: i32,
) -> String {
    assert!(room_size > 0);
    assert!(room_size_variance > 0);

    let mut f = FloorPlan::new_all_walls(width, height);

    let mut u = UFWithOwnKey::new();

    let mut all_non_boundary_cells = Vec::new();
    for x in 1..width - 1 {
        for y in 1..height - 1 {
            all_non_boundary_cells.push(f.from_xy(x, y));
        }
    }

    let mut rng = rand::rng();
    all_non_boundary_cells.shuffle(&mut rng);

    for cell in all_non_boundary_cells.iter() {
        if f.at(*cell) == GridElem::Wall {
            // if the number of distinct sets that empty neighbors belong to
            // isn't exactly 1, then knock it down
            let mut should_knock_down = false;
            let mut the_set = None;
            let mut num_empty_neighbors = 0;
            for neighbor in f
                .valid_neighbors(*cell)
                .iter()
                .filter(|n| f.at(**n) == GridElem::Empty)
            {
                num_empty_neighbors += 1;
                let set = u.find(*neighbor).unwrap();
                match the_set {
                    None => {
                        the_set = Some(set);
                    }
                    Some(already) => {
                        if already != set {
                            should_knock_down = true;
                            break;
                        }
                    }
                }
            }

            if num_empty_neighbors == 0 || num_empty_neighbors == 1 {
                should_knock_down = true;
            }

            if should_knock_down {
                f.set(*cell, GridElem::Empty);
                u.insert(*cell);
                for neighbor in f
                    .valid_neighbors(*cell)
                    .iter()
                    .filter(|n| f.at(**n) == GridElem::Empty)
                {
                    u.union(*cell, *neighbor);
                }
            }
        }
    }

    f.print();
    println!("\n");

    let mut room_count = 0;
    while room_count < num_rooms {
        let x = Rng::random_range(&mut rng, 0..width - 1) as i32;
        let y = Rng::random_range(&mut rng, 0..height - 1) as i32;

        let size_x =
            room_size - room_size_variance + Rng::random_range(&mut rng, 0..room_size_variance * 2);
        let size_y =
            room_size - room_size_variance + Rng::random_range(&mut rng, 0..room_size_variance * 2);

        if f.room_would_fit(x, y, x + size_x, y + size_y) {
            f.stamp_room(x, y, x + size_x, y + size_y);
            room_count += 1;
            f.set(
                f.from_xy((x + size_x / 2) as u32, (y + size_y / 2) as u32),
                GridElem::Exhibit,
            );
        }
    }

    loop {
        let mut dead_ends: Vec<GridIdx> = Vec::new();

        for g in f.all_cells() {
            if f.at(g) == GridElem::Empty {
                let num_wall_neighbors = f
                    .valid_neighbors(g)
                    .iter()
                    .filter(|n| f.at(**n) == GridElem::Wall)
                    .count();
                if num_wall_neighbors == 3 {
                    dead_ends.push(g);
                }
            }
        }

        if dead_ends.is_empty() {
            break;
        }

        for g in dead_ends {
            f.set(g, GridElem::Wall);
        }
    }

    f.print();
    println!("\n");

    "".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_generates() {
        generate_maze(16, 16, 3, 3, 1);
        generate_maze(40, 40, 15, 5, 3);

        assert_eq!(1, 0);
    }
}
