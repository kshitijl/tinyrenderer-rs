use crate::Args;
use crate::image::{BLUE, GOLD, GREY};
use crate::mesh::Mesh;
use crate::render::*;

use glam::{Mat3, USizeVec2, Vec2, Vec3, usizevec2, vec2, vec3};
use smallvec::SmallVec;
use std::collections::HashSet;
use std::f32;
use std::time::Duration;
use winit::keyboard::KeyCode;

pub enum ResolutionChangeAction {
    DoNothing,
    ChangeTo { x: u32, y: u32 },
}

struct Settings {
    rotate_objects: bool,
    move_light_with_camera: bool,
    topdown_camera: bool,
    draw_debug_lines: bool,
}

pub struct Camera {
    pub pos: Vec3,
    pub dir: Vec3,
    pub up: Vec3,
    mouse_x: f32,
    mouse_y: f32,
}

pub struct Spotlight {
    pub pos: Vec3,
    pub dir: Vec3,
}

#[derive(Debug)]
struct AABBXZ {
    min: Vec2,
    max: Vec2,
}

impl AABBXZ {
    // Returns 0 is point is inside the box.
    fn distance(&self, v: Vec3) -> f32 {
        let (x1, y1, x2, y2) = (self.min.x, self.min.y, self.max.x, self.max.y);
        let closest_x = v.x.clamp(x1, x2);
        let closest_yz = v.z.clamp(y1, y2);

        vec2(v.x - closest_x, v.z - closest_yz).length()
    }
}

enum Direction {
    Forward,
    Back,
    Right,
    Left,
}

#[derive(PartialEq)]
pub enum ObjectKind {
    Light,
    Exhibit,
    WallOrFloor,
}

pub struct Object {
    pub mesh: Mesh,
    pub pos: Vec3,
    pub angle_x: f32,
    pub angle_y: f32,
    pub scale: f32,
    pub color: Colorf,
    pub kind: ObjectKind,
    pub visible: bool,
}

mod fp {
    use super::*;

    #[derive(Copy, Debug, Clone, PartialEq)]
    pub enum GridElem {
        Wall,
        Empty,
        Exhibit,
    }
    pub struct FloorPlan {
        width: usize,
        height: usize,
        grid: Vec<GridElem>,
    }

    impl FloorPlan {
        pub fn from_string(s: &str) -> Self {
            let mut answer = Vec::new();
            let s = s.trim();

            let unique_widths: HashSet<usize> = s.lines().map(|line| line.len()).collect();

            if unique_widths.len() > 1 {
                panic!("grid not rectangular")
            }
            let width = *unique_widths.iter().next().unwrap();

            for c in s.chars() {
                let elem = match c {
                    'w' => Some(GridElem::Wall),
                    '.' => Some(GridElem::Empty),
                    'x' => Some(GridElem::Exhibit),
                    '\n' => None,
                    _ => panic!("unknown grid letter {}", c),
                };
                if let Some(e) = elem {
                    answer.push(e);
                }
            }

            let height = answer.len() / width;

            Self {
                width,
                height,
                grid: answer,
            }
        }

        pub fn height(&self) -> usize {
            self.height
        }

        pub fn width(&self) -> usize {
            self.width
        }

        pub fn valid_neighbors(&self, p: USizeVec2) -> SmallVec<[USizeVec2; 4]> {
            let mut answer = SmallVec::new();

            for (dx, dy) in [(1, 0), (-1i32, 0), (0, 1), (0, -1i32)] {
                let (neighbor_x, neighbor_y) = (p.x as i32 + dx, p.y as i32 + dy);
                if self.is_valid(neighbor_x, neighbor_y) {
                    answer.push(usizevec2(neighbor_x as usize, neighbor_y as usize));
                }
            }

            answer
        }

        pub fn at(&self, p: USizeVec2) -> GridElem {
            let y = self.height - p.y - 1;
            self.grid[y * self.width + p.x]
        }

        fn is_valid(&self, x: i32, y: i32) -> bool {
            x >= 0 && y >= 0 && x < self.width as i32 && y < self.height as i32
        }

        pub fn first_empty(&self) -> (usize, usize) {
            for y in 0..self.height {
                for x in 0..self.width {
                    if self.at(usizevec2(x, y)) == GridElem::Empty {
                        return (x, y);
                    }
                }
            }

            panic!("no empty");
        }
    }
}

use fp::{FloorPlan, GridElem};

struct Level {
    floor_plan: FloorPlan,
    grid_size: f32,
}

impl Level {
    fn new(floor_plan: FloorPlan, grid_size: f32) -> Self {
        Self {
            floor_plan,
            grid_size,
        }
    }

    fn grid2world(&self, x: usize, y: usize) -> Vec3 {
        vec3(
            (x as f32) * self.grid_size,
            -4.,
            (y as f32) * self.grid_size,
        )
    }

    fn world2grid(&self, v: Vec3) -> USizeVec2 {
        let x = (v.x / self.grid_size).round();
        let y = (v.z / self.grid_size).round();

        usizevec2(x as usize, y as usize)
    }

    fn aabb(&self, p: USizeVec2) -> AABBXZ {
        AABBXZ {
            min: vec2(
                (p.x as f32 - 0.5) * self.grid_size,
                (p.y as f32 - 0.5) * self.grid_size,
            ),
            max: vec2(
                (p.x as f32 + 0.5) * self.grid_size,
                (p.y as f32 + 0.5) * self.grid_size,
            ),
        }
    }
}

struct Player {
    pos: Vec3,
}

pub struct World {
    renderer: Renderer,
    settings: Settings,

    camera: Camera,

    light: Spotlight,
    light_object_idx: usize,

    player: Player,
    objects: Vec<Object>,
    level: Level,

    pub keys: HashSet<KeyCode>,
    pub first_pressed_this_frame: HashSet<KeyCode>,

    time_since_start: Duration,
    angle_time: Duration,
}

impl World {
    pub fn new(args: &Args) -> Self {
        let mut objects = Vec::new();

        let g = FloorPlan::from_string(
            r#"
wwwwwww
w...www
w..w.ww
w..w..w
w..w..w
w..w..w
w.....w
w...x.w
w.....w
wwwwwww"#,
        );

        // //         let g = FloorPlan::from_string(
        // //             r#"
        // wwwwwwwwww
        // w.wwwwwwww
        // wwwwwwwwww
        // "#,
        //         );

        for y in 0..g.height() {
            for x in 0..g.width() {
                println!("({},{}) {:?} ", x, y, g.at(usizevec2(x, y)));
            }
            println!("");
        }

        println!("exhibit at 4,2 {:?}", g.at(usizevec2(4, 2)));

        let level = Level::new(g, 2.);
        let wall_color = Colorf(vec3(0.5, 0.5, 0.5));
        let wall_color1 = Colorf(vec3(1., 0., 0.));
        let green = Colorf(vec3(0., 1., 0.));
        let blue = Colorf(vec3(0., 0., 1.));
        let yellow = Colorf(vec3(1., 1., 0.));
        let exhibits_color = Colorf(vec3(1., 155. / 255., 0.));

        let make_floor = |x, y| {
            let y_offset = -3.;

            Object {
                mesh: Mesh::wall(),
                pos: level.grid2world(x, y) + vec3(0., y_offset, 0.),
                angle_x: -90f32.to_radians(),
                angle_y: 0.,
                scale: 1.,
                color: wall_color,
                kind: ObjectKind::WallOrFloor,
                visible: true,
            }
        };

        let make_wall = |x, y, facing, y_offset| {
            let (angle_y, x_offset, z_offset, _color) = match facing {
                (-1, 0) => (-90f32.to_radians(), -1., 0., wall_color1),
                (1, 0) => (90f32.to_radians(), 1., 0., green),
                (0, -1) => (180f32.to_radians(), 0., -1., blue),
                (0, 1) => (0f32.to_radians(), 0., 1., yellow),
                _ => panic!("weird facing {:?}", facing),
            };

            Object {
                mesh: Mesh::wall(),
                pos: level.grid2world(x, y) + vec3(x_offset, y_offset, z_offset),
                angle_x: 0.,
                angle_y,
                scale: 1.,
                color: wall_color,
                kind: ObjectKind::WallOrFloor,
                visible: true,
            }
        };

        let mut exhibit_idx = 0;

        for x in 0..level.floor_plan.width() {
            for y in 0..level.floor_plan.height() {
                let mesh: Mesh;
                let angle_x = 0.;
                let object_color: Colorf;
                let y_offset = 0.;

                match level.floor_plan.at(usizevec2(x, y)) {
                    GridElem::Wall => {
                        for neighbor in level.floor_plan.valid_neighbors(usizevec2(x, y)) {
                            if level.floor_plan.at(neighbor) == GridElem::Empty {
                                for y_offset in [-2., 0., 2.] {
                                    if let Some(debug_wall) = &args.wall_model_debug {
                                        let mut model =
                                            Mesh::from_file(debug_wall.as_str()).unwrap();
                                        model.normalize();
                                        objects.push(Object {
                                            mesh: model,
                                            pos: level.grid2world(x, y).with_y(-7.),
                                            angle_x: 0.,
                                            angle_y: 0.,
                                            scale: 1.,
                                            color: wall_color,
                                            kind: ObjectKind::WallOrFloor,
                                            visible: true,
                                        });
                                    } else {
                                        objects.push(make_wall(
                                            x,
                                            y,
                                            (
                                                neighbor.x as i32 - x as i32,
                                                neighbor.y as i32 - y as i32,
                                            ),
                                            y_offset,
                                        ));
                                    }
                                }
                            }
                        }
                    }
                    GridElem::Empty => {
                        objects.push(make_floor(x, y));
                    }
                    GridElem::Exhibit => {
                        let mut model = Mesh::from_file(args.exhibit_model.as_str()).unwrap();
                        model.normalize();
                        mesh = model;
                        object_color = exhibits_color;
                        let pos = level.grid2world(x, y) + vec3(0., y_offset, 0.);
                        assert!(level.world2grid(pos) == usizevec2(x, y));

                        log::info!(
                            "instantiating exhibit at {}, world {}, aabb {:?}",
                            usizevec2(x, y),
                            pos,
                            level.aabb(usizevec2(x, y))
                        );
                        objects.push(Object {
                            mesh,
                            pos,
                            angle_x,
                            angle_y: 0.,
                            scale: 1.,
                            color: object_color,
                            kind: ObjectKind::Exhibit,
                            visible: true,
                        });
                        exhibit_idx = objects.len() - 1;
                        objects.push(make_floor(x, y));
                    }
                }
            }
        }

        objects.push(Object {
            mesh: objects[exhibit_idx].mesh.clone(),
            pos: vec3(0., 0., 0.),
            angle_x: 0.,
            angle_y: 0.,
            scale: 0.3,
            color: Colorf(vec3(1., 1., 1.)),
            kind: ObjectKind::Light,
            visible: false,
        });
        let light_object_idx = objects.len() - 1;

        let (x_empty, y_empty) = level.floor_plan.first_empty();

        let player = Player {
            pos: level.grid2world(x_empty, y_empty) + vec3(0.2, 0.4, 0.2),
        };

        let initial_camera_pos = player.pos;

        log::info!(
            "initial player position in world is {}, in grid is {}",
            player.pos,
            level.world2grid(player.pos)
        );

        let camera_dir = Vec3::ZERO;

        let light = Spotlight {
            pos: player.pos,
            dir: camera_dir,
        };

        let renderer = Renderer::new(args.canvas_size);

        Self {
            renderer,
            objects,
            player,
            camera: Camera {
                pos: initial_camera_pos,
                dir: camera_dir,
                up: vec3(0., 1., 0.).normalize(),
                mouse_x: 180f32.to_radians(),
                mouse_y: 0.,
            },
            keys: HashSet::new(),
            first_pressed_this_frame: HashSet::new(),
            light,
            light_object_idx,
            time_since_start: Duration::from_secs(0),
            angle_time: Duration::from_secs(0),
            settings: Settings {
                rotate_objects: false,
                move_light_with_camera: true,
                topdown_camera: false,
                draw_debug_lines: false,
            },
            level,
        }
    }

    pub fn width(&self) -> usize {
        self.renderer.width
    }

    pub fn camera_mouse(&mut self, dx: f64, dy: f64) {
        let (dx, dy) = (dx as f32, dy as f32);
        let scale = -1e-3;
        self.camera.mouse_x = (self.camera.mouse_x + dx * scale).rem_euclid(2. * f32::consts::PI);
        self.camera.mouse_y += dy * scale;
    }

    fn move_(&mut self, dir: Direction) {
        let speed = 0.1;

        let forward = self.camera.dir.with_y(0.).normalize();
        let right = forward.cross(self.camera.up);

        // let forward = vec3(0., 0., -1.);
        // let right = vec3(1., 0., 0.);

        let desired_pos = match dir {
            Direction::Forward => self.camera.pos + forward * speed,
            Direction::Back => self.camera.pos - forward * speed,
            Direction::Right => self.camera.pos + right * speed,
            Direction::Left => self.camera.pos - right * speed,
        };

        /*
        Super dumb and simple collision detection.

        High level: prevent getting too close to walls, exhibits and guards,
        where "too close" is defined by some epsilon.

        Medium level: find the distance between the desired position and each
        possible collider. If any distance is below epsilon, reject motion.

        TODO what if guards trap the player lol?
        TODO allow motion all the way up to the boundary but not beyond

        Details:
         - iterate over possible colliders: nearby walls and exhibits; all guards
         - get their AABB in the X-Z plane
         - find the distance from desired pos to each AABB
         - if any distance < epsilon then reject motion
        */
        let eps = 0.1;

        log::info!(
            "current: {}. desired: {}. current grid loc: {}. desired grid loc: {}",
            self.camera.pos,
            desired_pos,
            self.level.world2grid(self.camera.pos),
            self.level.world2grid(desired_pos)
        );

        let mut current_min_distance = f32::MAX;
        let mut desired_min_distance = f32::MAX;
        let current_grid_pos = self.level.world2grid(self.camera.pos);
        for neighbor in self.level.floor_plan.valid_neighbors(current_grid_pos) {
            match self.level.floor_plan.at(neighbor) {
                GridElem::Wall | GridElem::Exhibit => {
                    let aabb = self.level.aabb(neighbor);
                    let desired_distance = aabb.distance(desired_pos);
                    let current_distance = aabb.distance(self.camera.pos);

                    current_min_distance = f32::min(current_distance, current_min_distance);
                    desired_min_distance = f32::min(desired_distance, desired_min_distance);
                }
                GridElem::Empty => {
                    // do nothing, always allow
                }
            }
        }

        if desired_min_distance < eps && desired_min_distance < current_min_distance {
            // do not allow
        } else {
            // log::info!("movement was allowed. new loc is {}", desired_pos);
            self.camera.pos = desired_pos;
        }

        // log::info!(
        //     "now at grid {:?}, world {}",
        //     self.level.world2grid(self.camera.pos),
        //     self.camera.pos
        // );
    }

    fn is_key_down(&self, key: KeyCode) -> bool {
        self.keys.contains(&key)
    }

    fn was_key_pressed(&self, key: KeyCode) -> bool {
        self.first_pressed_this_frame.contains(&key)
    }

    fn handle_keys(&mut self) -> ResolutionChangeAction {
        let mut answer = ResolutionChangeAction::DoNothing;

        if self.is_key_down(KeyCode::KeyW) {
            self.move_(Direction::Forward);
        }

        if self.is_key_down(KeyCode::KeyS) {
            self.move_(Direction::Back);
        }

        if self.is_key_down(KeyCode::KeyA) {
            self.move_(Direction::Left);
        }

        if self.is_key_down(KeyCode::KeyD) {
            self.move_(Direction::Right);
        }

        if self.was_key_pressed(KeyCode::Digit1) {
            self.renderer.render_settings.wireframe =
                self.renderer.render_settings.wireframe.next();
        }
        if self.was_key_pressed(KeyCode::Digit3) {
            self.settings.move_light_with_camera = !self.settings.move_light_with_camera;
        }
        if self.was_key_pressed(KeyCode::Digit4) {
            self.settings.rotate_objects = !self.settings.rotate_objects;
        }
        if self.was_key_pressed(KeyCode::Digit5) {
            let (new_x, new_mode) = match self.renderer.render_settings.split_screen_mode {
                SplitScreenMode::Normal => (self.width() * 2, SplitScreenMode::Split),
                SplitScreenMode::Split => (self.width(), SplitScreenMode::Normal),
            };

            answer = ResolutionChangeAction::ChangeTo {
                x: new_x as u32,
                y: self.width() as u32,
            };
            self.renderer.render_settings.split_screen_mode = new_mode;
        }
        if self.was_key_pressed(KeyCode::Digit6) {
            self.settings.draw_debug_lines = !self.settings.draw_debug_lines;
        }
        if self.was_key_pressed(KeyCode::Digit7) {
            self.settings.topdown_camera = !self.settings.topdown_camera;

            self.objects[self.light_object_idx].visible =
                self.settings.topdown_camera || !self.settings.move_light_with_camera;
        }

        self.first_pressed_this_frame.clear();
        answer
    }

    fn animate_objects(&mut self, since_last_frame: Duration) {
        if self.settings.rotate_objects {
            self.angle_time += since_last_frame;
        }

        for object in self.objects.iter_mut() {
            let angle = self.angle_time.as_secs_f32();
            match object.kind {
                ObjectKind::Exhibit => {
                    object.angle_y = angle;
                }
                ObjectKind::Light | ObjectKind::WallOrFloor => {
                    // do nothing
                }
            }
        }

        if self.settings.move_light_with_camera {
            let t = self.time_since_start.as_secs_f32();

            // self.light.pos =
            //     self.camera.pos + vec3(0.1 * f32::sin(t), -0.6 + 0.1 * f32::cos(0.7 * t), -0.1);

            if self.settings.topdown_camera {
                self.light.dir = Mat3::from_rotation_y(self.camera.mouse_x) * vec3(0., 0., -1.);
                self.light.pos = self.camera.pos.with_y(-3.6);
            } else {
                self.light.dir = self.camera.dir;
                self.light.pos = self.camera.pos.with_y(self.camera.pos.y - 0.6);
            }

            self.objects[self.light_object_idx].pos = self.light.pos;
        }
    }

    fn update_camera(&mut self) {
        let (y1, y2) = if self.settings.topdown_camera {
            (-f32::consts::PI / 2.1, -f32::consts::PI / 2.6)
        } else {
            (-f32::consts::PI / 2.1, f32::consts::PI / 3.)
        };
        self.camera.mouse_y = self.camera.mouse_y.clamp(y1, y2);

        self.camera.dir = Mat3::from_rotation_y(self.camera.mouse_x)
            * Mat3::from_rotation_x(self.camera.mouse_y)
            * vec3(0., 0., -1.);

        if self.settings.topdown_camera {
            self.camera.pos.y = 7.;
        } else {
            self.camera.pos.y = -3.;
        }
    }

    pub fn update(
        &mut self,
        since_last_frame: Duration,
        since_start: Duration,
    ) -> ResolutionChangeAction {
        self.time_since_start = since_start;

        let answer = self.handle_keys();
        self.update_camera();
        self.animate_objects(since_last_frame);

        answer
    }

    pub fn draw(&mut self, frame: &mut [u8]) -> RenderingResult {
        if self.settings.draw_debug_lines {
            let f = |v: Vec2| Vec3::new(v.x, -7., v.y);
            for neighbor in self
                .level
                .floor_plan
                .valid_neighbors(self.level.world2grid(self.camera.pos))
                .iter()
            {
                let color = match self.level.floor_plan.at(*neighbor) {
                    GridElem::Wall => GREY,
                    GridElem::Empty => BLUE,
                    GridElem::Exhibit => GOLD,
                };
                let aabb = self.level.aabb(*neighbor);

                let p1 = aabb.min;
                let p2 = vec2(aabb.min.x, aabb.max.y);
                let p3 = aabb.max;
                let p4 = vec2(aabb.max.x, aabb.min.y);

                for [a, b] in [
                    [f(p1), f(p2)],
                    [f(p2), f(p3)],
                    [f(p3), f(p4)],
                    [f(p4), f(p1)],
                ] {
                    self.renderer.debug_draw_line_in_world_space(a, b, color);
                }
            }
        }

        self.renderer
            .draw(&self.light, &self.camera, &self.objects, frame)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_parses_grid() {
        let g = FloorPlan::from_string(
            r#"
wwwwwwwww
w...wwwww
w.ww.wwww
w.ww...ww
w.ww...ww
w.ww...ww
w.......w
w...x...w
w.......w
wwwwwwwww"#,
        );

        assert_eq!(g.width(), 9);
        assert_eq!(g.height(), 10);
        assert_eq!(g.first_empty(), (1, 1));
    }

    #[test]
    fn it_computes_aabb_distances() {
        let aabb = AABBXZ {
            min: vec2(1.0, 1.0),
            max: vec2(3.0, 3.0),
        };

        assert_eq!(aabb.distance(vec3(1., 1., 1.)), 0.);
        assert_eq!(aabb.distance(vec3(1., 100., 1.)), 0.);
        assert!((aabb.distance(vec3(0.5, 100., 0.5)) - 0.5f32.sqrt()).abs() < 1e-6);
        assert_eq!(aabb.distance(vec3(1.5, 100., 0.5)), 0.5);
        assert_eq!(aabb.distance(vec3(1.5, 100., 2.7)), 0.);
        assert_eq!(aabb.distance(vec3(1.5, 100., 2.75)), 0.);
        assert_eq!(aabb.distance(vec3(2.0, 100., 6.)), 3.);
    }
}
