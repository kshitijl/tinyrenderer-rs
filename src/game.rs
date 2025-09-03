use crate::Args;
use crate::mesh::Mesh;
use crate::render::*;

use glam::{Mat3, Vec3, vec3};
use std::collections::HashSet;
use std::f32;
use std::time::Duration;
use winit::keyboard::KeyCode;

pub enum ResolutionChangeAction {
    DoNothing,
    ChangeTo { x: u32, y: u32 },
}

struct MovementSettings {
    rotate_objects: bool,
    move_light_with_camera: bool,
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

enum Direction {
    Forward,
    Back,
    Right,
    Left,
}

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
}

#[derive(Copy, Clone, PartialEq)]
enum GridElem {
    Wall,
    Empty,
    Exhibit,
}

struct FloorPlan {
    width: usize,
    height: usize,
    grid: Vec<GridElem>,
}

impl FloorPlan {
    fn from_string(s: &str) -> Self {
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

    fn at(&self, x: usize, y: usize) -> GridElem {
        self.grid[y * self.width + x]
    }

    fn is_valid(&self, x: i32, y: i32) -> bool {
        x >= 0 && y >= 0 && x < self.width as i32 && y < self.height as i32
    }

    fn first_empty(&self) -> (usize, usize) {
        for y in 0..self.height {
            for x in 0..self.width {
                if self.at(x, y) == GridElem::Empty {
                    return (x, y);
                }
            }
        }

        panic!("no empty");
    }
}

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
        vec3(x as f32 * self.grid_size, -4., y as f32 * self.grid_size)
    }

    fn world2grid(&self, v: Vec3) -> (usize, usize) {
        let x = (v.x / self.grid_size).round();
        let y = (v.z / self.grid_size).round();

        (x as usize, y as usize)
    }
}

pub struct World {
    renderer: Renderer,
    movement_settings: MovementSettings,

    camera: Camera,

    light: Spotlight,
    light_object_idx: usize,

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
            }
        };

        for x in 0..level.floor_plan.width {
            for y in 0..level.floor_plan.height {
                let mesh: Mesh;
                let angle_x = 0.;
                let object_color: Colorf;
                let y_offset = 0.;

                match level.floor_plan.at(x, y) {
                    GridElem::Wall => {
                        for (dx, dy) in [(1, 0), (-1i32, 0), (0, 1), (0, -1i32)] {
                            let (neighbor_x, neighbor_y) = (x as i32 + dx, y as i32 + dy);
                            if level.floor_plan.is_valid(neighbor_x, neighbor_y)
                                && level
                                    .floor_plan
                                    .at(neighbor_x as usize, neighbor_y as usize)
                                    == GridElem::Empty
                            {
                                for y_offset in [-2., 0., 2.] {
                                    objects.push(make_wall(x, y, (dx, dy), y_offset));
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

                        objects.push(Object {
                            mesh,
                            pos: level.grid2world(x, y) + vec3(0., y_offset, 0.),
                            angle_x,
                            angle_y: 0.,
                            scale: 1.,
                            color: object_color,
                            kind: ObjectKind::Exhibit,
                        });

                        objects.push(make_floor(x, y));
                    }
                }
            }
        }

        objects.push(Object {
            mesh: objects[0].mesh.clone(),
            pos: vec3(0., 0., 0.),
            angle_x: 0.,
            angle_y: 0.,
            scale: 0.3,
            color: Colorf(vec3(1., 1., 1.)),
            kind: ObjectKind::Light,
        });
        let light_object_idx = objects.len() - 1;

        let (x_empty, y_empty) = level.floor_plan.first_empty();
        let initial_camera_pos = level.grid2world(x_empty, y_empty) + vec3(0., 0.2, 0.);
        let camera_dir = Vec3::ZERO;

        let light = Spotlight {
            pos: initial_camera_pos,
            dir: camera_dir,
        };

        let renderer = Renderer::new(args.canvas_size);

        Self {
            renderer,
            objects,
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
            movement_settings: MovementSettings {
                rotate_objects: false,
                move_light_with_camera: true,
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
        self.camera.mouse_y =
            (self.camera.mouse_y + dy * scale).clamp(-f32::consts::PI / 3., f32::consts::PI / 3.);
    }

    fn move_(&mut self, dir: Direction) {
        let speed = 0.1;
        let forward = self.camera.dir.with_y(0.).normalize();
        let right = forward.cross(self.camera.up);

        let new_pos = match dir {
            Direction::Forward => self.camera.pos + forward * speed,
            Direction::Back => self.camera.pos - forward * speed,
            Direction::Right => self.camera.pos + right * speed,
            Direction::Left => self.camera.pos - right * speed,
        };

        let (grid_x, grid_y) = self.level.world2grid(new_pos);
        match self.level.floor_plan.at(grid_x, grid_y) {
            GridElem::Wall => {
                // don't allow
            }
            GridElem::Exhibit => {
                // don't allow
            }
            GridElem::Empty => {
                self.camera.pos = new_pos;
            }
        }

        log::info!(
            "now at grid {:?}, world {}",
            self.level.world2grid(self.camera.pos),
            self.camera.pos
        );
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
            self.movement_settings.move_light_with_camera =
                !self.movement_settings.move_light_with_camera;
        }
        if self.was_key_pressed(KeyCode::Digit4) {
            self.movement_settings.rotate_objects = !self.movement_settings.rotate_objects;
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

        self.first_pressed_this_frame.clear();
        answer
    }

    fn animate_objects(&mut self, since_last_frame: Duration) {
        if self.movement_settings.rotate_objects {
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

        if self.movement_settings.move_light_with_camera {
            let t = self.time_since_start.as_secs_f32();

            self.light.pos =
                self.camera.pos + vec3(0.1 * f32::sin(t), -0.6 + 0.1 * f32::cos(0.7 * t), -0.1);
            self.light.dir = self.camera.dir;
            self.objects[self.light_object_idx].pos = self.light.pos;
        }
    }

    fn update_camera(&mut self) {
        self.camera.dir = Mat3::from_rotation_y(self.camera.mouse_x)
            * Mat3::from_rotation_x(self.camera.mouse_y)
            * vec3(0., 0., -1.);
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

        assert_eq!(g.width, 9);
        assert_eq!(g.height, 10);
        assert_eq!(g.first_empty(), (1, 1));
    }
}
