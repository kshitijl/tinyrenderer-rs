use crate::Args;
use crate::audio::{self, AudioSystem};
use crate::image::{BLACK, BLUE, GOLD, GREY};
use crate::mesh::Mesh;
use crate::render::*;
use glam::{Mat3, Vec2, Vec3, vec2, vec3};
use mazegen::{FloorPlan, GridElem, GridIdx};
use rand::rngs::ThreadRng;
use rand::seq::IndexedRandom;
use rand::{self, Rng};
use std::collections::{HashMap, HashSet};
use std::f32;
use std::time::Duration;
use winit::keyboard::KeyCode;

mod mazegen;

pub enum ResolutionChangeAction {
    DoNothing,
    ChangeTo { x: u32, y: u32 },
}

struct Settings {
    rotate_objects: bool,
    draw_debug_lines: bool,
}

pub struct Camera {
    pub pos: Vec3,
    pub dir: Vec3,
    pub up: Vec3,
    mouse_x: f32,
    mouse_y: f32,
}

#[derive(Copy, Clone)]
pub struct Flashlight {
    pub pos: Vec3,
    pub dir: Vec3,
}

#[derive(Copy, Clone)]
pub struct Spotlight {
    pub pos: Vec3,
    pub color: Colorf,
}

#[derive(Debug)]
struct AABBXZ {
    min: Vec2,
    max: Vec2,
}

impl AABBXZ {
    // Returns 0 if point is inside the box.
    fn distance(&self, v: Vec3) -> f32 {
        let (x1, y1, x2, y2) = (self.min.x, self.min.y, self.max.x, self.max.y);
        let closest_x = v.x.clamp(x1, x2);
        let closest_yz = v.z.clamp(y1, y2);

        vec2(v.x - closest_x, v.z - closest_yz).length()
    }

    fn normal(&self, towards: &Vec3) -> Vec3 {
        // TODO figure out what to do when [towards] is inside the AABB
        if towards.x < self.min.x {
            vec3(-1., 0., 0.)
        } else if towards.x > self.max.x {
            vec3(1., 0., 0.)
        } else if towards.z < self.min.y {
            vec3(0., 0., -1.)
        } else if towards.z > self.max.y {
            vec3(0., 0., 1.)
        } else {
            // lol
            let center = vec3(
                (self.min.x + self.max.x) / 2.,
                towards.y,
                (self.min.y + self.max.y) / 2.,
            );
            towards - center
        }
    }
}

enum Direction {
    Forward,
    Back,
    Right,
    Left,
}

#[derive(PartialEq, Debug)]
pub enum ObjectKind {
    Light,
    Exhibit { hiddenness: f32 },
    WallOrFloor,
    Guard,
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

    fn grididx2world(&self, g: GridIdx) -> Vec3 {
        let (x, y) = self.floor_plan.to_xy(g);
        self.grid2world(x, y)
    }

    fn grid2world(&self, x: u32, y: u32) -> Vec3 {
        vec3(
            (x as f32) * self.grid_size,
            -4.,
            (y as f32) * self.grid_size,
        )
    }

    fn world2grid(&self, v: Vec3) -> GridIdx {
        let x = (v.x / self.grid_size).round();
        let y = (v.z / self.grid_size).round();

        self.floor_plan.from_xy(x as u32, y as u32)
    }

    fn aabb(&self, g: GridIdx) -> AABBXZ {
        let (x, y) = self.floor_plan.to_xy(g);
        AABBXZ {
            min: vec2(
                (x as f32 - 0.5) * self.grid_size,
                (y as f32 - 0.5) * self.grid_size,
            ),
            max: vec2(
                (x as f32 + 0.5) * self.grid_size,
                (y as f32 + 0.5) * self.grid_size,
            ),
        }
    }
}

struct Player {
    pos: Vec3,
}

#[derive(Copy, Clone, PartialEq)]
enum ViewMode {
    Topdown,
    Fps { last_topdown_y: f32 },
}

#[derive(Debug)]
struct ObjectIdx(usize);

#[derive(Debug)]
struct SpotlightIdx(usize);

#[derive(Debug, Clone, Copy)]
enum GridDir {
    XPlus,
    XMinus,
    ZPlus,
    ZMinus,
}

impl GridDir {
    const ALL_DIRS: [GridDir; 4] = [
        GridDir::XPlus,
        GridDir::XMinus,
        GridDir::ZPlus,
        GridDir::ZMinus,
    ];
    fn all() -> &'static [GridDir; 4] {
        &Self::ALL_DIRS
    }

    fn flip(&self) -> Self {
        match *self {
            GridDir::XPlus => GridDir::XMinus,
            GridDir::XMinus => GridDir::XPlus,
            GridDir::ZPlus => GridDir::ZMinus,
            GridDir::ZMinus => GridDir::ZPlus,
        }
    }
    fn to_world_dir(&self) -> Vec3 {
        match *self {
            GridDir::XPlus => vec3(1., 0., 0.),
            GridDir::XMinus => vec3(-1., 0., 0.),
            GridDir::ZPlus => vec3(0., 0., 1.),
            GridDir::ZMinus => vec3(0., 0., -1.),
        }
    }
    fn to_world_angle(&self) -> f32 {
        match *self {
            GridDir::XPlus => 90f32.to_radians(),
            GridDir::XMinus => -90f32.to_radians(),
            GridDir::ZPlus => 0.,
            GridDir::ZMinus => 180f32.to_radians(),
        }
    }

    fn random(rng: &mut ThreadRng) -> Self {
        *Self::all().choose(rng).unwrap()
    }
}

#[derive(Debug, Copy, Clone)]
enum GuardState {
    Alarmed,
    Beat { facing: GridDir },
}

impl GuardState {
    fn beat_facing_random(rng: &mut ThreadRng) -> Self {
        Self::Beat {
            facing: GridDir::random(rng),
        }
    }
}

#[derive(Debug)]
struct Guard {
    idx: ObjectIdx,
    spotlight: SpotlightIdx,
    state: GuardState,
}

pub struct World {
    renderer: Renderer,
    audio: AudioSystem,
    settings: Settings,

    camera: Camera,
    vm: ViewMode,

    light: Flashlight,
    light_object_idx: usize,
    spotlights: Vec<Spotlight>,

    player: Player,
    guards: Vec<Guard>,
    objects: Vec<Object>,
    level: Level,
    g2o: HashMap<GridIdx, ObjectIdx>,

    pub keys: HashSet<KeyCode>,
    pub first_pressed_this_frame: HashSet<KeyCode>,

    time_since_start: Duration,
    rng: ThreadRng,
}

impl World {
    pub fn new(args: &Args, audio: AudioSystem) -> Self {
        let mut objects = Vec::new();

        // let g = FloorPlan::from_string(
        //     r#"
        // wwwwwwwwwwwww
        // w.....wwwwwww
        // w..x..wwwwwww
        // w....wwwwwwww
        // w..ww......ww
        // w..w....x...w
        // w..w........w
        // w..w....x...w
        // w...........w
        // w.......x...w
        // w...........w
        // wwwwwwwwwwwww"#,
        // );

        let g = FloorPlan::generate(70, 20, 12, 5, 3, 8);

        g.print();

        let theme_colors = vec![
            chex(0xea369e),
            chex(0xea3e7a),
            chex(0xfdf952),
            chex(0xec6734),
            chex(0xeb5943),
            chex(0xeb4d59),
            chex(0x68ded3),
            chex(0x60cee6),
            chex(0x58bff9),
            chex(0x5fc697),
            Colorf(vec3(0.3, 0.3, 0.3)),
        ];
        fn chex(hex: u32) -> Colorf {
            let r = ((hex >> 16) & 0xff) as f32 / 255.;
            let g = ((hex >> 8) & 0xff) as f32 / 255.;
            let b = (hex & 0xff) as f32 / 255.;
            Colorf(vec3(r, g, b))
        }

        let mut rng = rand::rng();
        let floor_black = *theme_colors.choose(&mut rng).unwrap();
        let floor_white = Colorf(vec3(0.9, 0.9, 0.9));
        let mut wall_color = theme_colors.choose(&mut rng).unwrap();
        // let exhibits_color = Colorf(vec3(1., 155. / 255., 0.));

        let level = Level::new(g, 2.);
        let make_floor = |x, y| {
            let color = if (x + y) % 2 == 0 {
                floor_white
            } else {
                floor_black
            };
            let y_offset = -3.;

            Object {
                mesh: Mesh::wall(),
                pos: level.grid2world(x, y) + vec3(0., y_offset, 0.),
                angle_x: -90f32.to_radians(),
                angle_y: 0.,
                scale: 1.,
                color,
                kind: ObjectKind::WallOrFloor,
                visible: true,
            }
        };

        let make_wall = |x, y, facing, y_offset, wall_color: Colorf| {
            let (angle_y, x_offset, z_offset) = match facing {
                (-1, 0) => (-90f32.to_radians(), -1., 0.),
                (1, 0) => (90f32.to_radians(), 1., 0.),
                (0, -1) => (180f32.to_radians(), 0., -1.),
                (0, 1) => (0f32.to_radians(), 0., 1.),
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

        let mut exhibit_models = args.exhibit_models.iter().cycle();

        let mut g2o = HashMap::new();

        for x in 0..level.floor_plan.width() {
            for y in 0..level.floor_plan.height() {
                let angle_x = 0.;
                let y_offset = 0.;
                let g = level.floor_plan.from_xy(x, y);
                if (y * level.floor_plan.width() + x) % 100 == 0 {
                    wall_color = theme_colors.choose(&mut rng).unwrap();
                }
                match level.floor_plan.at(g) {
                    GridElem::Wall => {
                        for neighbor in level.floor_plan.valid_neighbors_no_diagonals(g) {
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
                                            color: *wall_color,
                                            kind: ObjectKind::WallOrFloor,
                                            visible: true,
                                        });
                                    } else {
                                        let (nx, ny) = level.floor_plan.to_xy(neighbor);
                                        objects.push(make_wall(
                                            x,
                                            y,
                                            (nx as i32 - x as i32, ny as i32 - y as i32),
                                            y_offset,
                                            *wall_color,
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
                        let mut model =
                            Mesh::from_file(exhibit_models.next().unwrap().as_str()).unwrap();
                        model.normalize();

                        let color = *theme_colors.choose(&mut rng).unwrap();
                        let pos = level.grid2world(x, y) + vec3(0., y_offset, 0.);
                        assert!(level.world2grid(pos) == level.floor_plan.from_xy(x, y));
                        assert!(level.floor_plan.to_xy(level.world2grid(pos)) == (x, y));

                        log::info!(
                            "instantiating exhibit at {:?}, world {}, aabb {:?}",
                            (x, y),
                            pos,
                            level.aabb(level.floor_plan.from_xy(x, y))
                        );
                        objects.push(Object {
                            mesh: model,
                            pos,
                            angle_x,
                            angle_y: 0.,
                            scale: 1.,
                            color,
                            kind: ObjectKind::Exhibit { hiddenness: 1.0 },
                            visible: true,
                        });
                        g2o.insert(level.floor_plan.from_xy(x, y), ObjectIdx(objects.len() - 1));
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
            visible: true,
        });
        let light_object_idx = objects.len() - 1;

        let mut guards = Vec::new();
        let mut spotlights = Vec::new();

        {
            let guard_spotlight_color = vec3(0.2, 0.2, 0.2);
            let guard_color = Colorf(vec3(1.0, 1., 1.));
            for _ in 0..args.num_guards {
                let pos = level
                    .grididx2world(level.floor_plan.random_empty(&mut rng))
                    .with_y(-4.5);

                let mut guard_mesh = Mesh::from_file(&args.guard_model).unwrap();
                guard_mesh.normalize();

                objects.push(Object {
                    mesh: guard_mesh,
                    pos,
                    angle_x: 0.,
                    angle_y: 0.,
                    scale: 2.6,
                    color: guard_color,
                    kind: ObjectKind::Guard,
                    visible: true,
                });

                spotlights.push(Spotlight {
                    pos: pos,
                    color: Colorf(guard_spotlight_color),
                });

                guards.push(Guard {
                    idx: ObjectIdx(objects.len() - 1),
                    spotlight: SpotlightIdx(spotlights.len() - 1),
                    state: GuardState::beat_facing_random(&mut rng),
                });
            }
        }

        let (x_empty, y_empty) = level.floor_plan.to_xy(level.floor_plan.first_empty());

        let player = Player {
            pos: level.grid2world(x_empty, y_empty) + vec3(0.2, 0.4, 0.2),
        };

        let initial_camera_pos = player.pos;

        log::info!(
            "initial player position in world is {}, in grid is {:?}",
            player.pos,
            level.floor_plan.to_xy(level.world2grid(player.pos))
        );

        let camera_dir = Vec3::ZERO;

        let light = Flashlight {
            pos: player.pos,
            dir: camera_dir,
        };

        let renderer = Renderer::new(args.canvas_size);

        Self {
            renderer,
            audio,
            objects,
            player,
            guards,
            vm: ViewMode::Topdown,
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
            spotlights,
            time_since_start: Duration::from_secs(0),
            settings: Settings {
                rotate_objects: false,
                draw_debug_lines: false,
            },
            level,
            g2o,
            rng,
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

    fn move_(&mut self, dir: Direction, since_last_frame: Duration) {
        let speed = 10. * since_last_frame.as_secs_f32();

        let forward = self.camera.dir.with_y(0.).normalize();
        let right = forward.cross(self.camera.up);

        let desired_pos = match dir {
            Direction::Forward => self.player.pos + forward * speed,
            Direction::Back => self.player.pos - forward * speed,
            Direction::Right => self.player.pos + right * speed,
            Direction::Left => self.player.pos - right * speed,
        };

        /*
        Simple collision detection & response that makes player slide along
        walls.

        For each nearby collider, figure out if desired pos is inside it.
        If yes, find the normal from the collider to initial pos.
        Also find how far inside the aabb the desired pos is.
        Move the desired pos out along the normal by that dist.
        Now the new desired pos should be outside the aabb of the collider.
        Take this desired pos and continue with next collider.
        At the end of this process we should be colliding with nothing.
        Move player there.
        */

        let eps = 0.1;
        let current_world_pos = self.player.pos;
        let current_grid_pos = self.player_grid_pos();
        let mut final_pos = desired_pos;

        for neighbor in self
            .level
            .floor_plan
            .valid_neighbors_no_diagonals(current_grid_pos)
        {
            match self.level.floor_plan.at(neighbor) {
                GridElem::Wall | GridElem::Exhibit => {
                    let aabb = self.level.aabb(neighbor);
                    let desired_distance = aabb.distance(final_pos);

                    if desired_distance <= eps {
                        let normal = aabb.normal(&current_world_pos);

                        // TODO replace this loop with a single calculation to
                        // how deep inside we are
                        while aabb.distance(final_pos) <= eps {
                            final_pos += eps * normal;
                        }
                    }
                }
                GridElem::Empty => {
                    // do nothing, always allow
                }
            }
        }
        self.player.pos = final_pos;
    }

    fn is_key_down(&self, key: KeyCode) -> bool {
        self.keys.contains(&key)
    }

    fn was_key_pressed(&self, key: KeyCode) -> bool {
        self.first_pressed_this_frame.contains(&key)
    }

    #[allow(dead_code)]
    fn log_player_position(&self) {
        let a = self.player.pos;
        let p = self.level.world2grid(self.player.pos);
        let (x, y) = self.level.floor_plan.to_xy(p);
        let b = self.level.grid2world(x, y);
        log::info!(
            "player at {}. on grid {:?}, grid xy {:?}, back on world {}, aabb {:?}",
            a,
            p,
            self.level.floor_plan.to_xy(p),
            b,
            self.level.aabb(p)
        );
    }

    fn player_grid_pos(&self) -> GridIdx {
        self.level.world2grid(self.player.pos)
    }

    fn examine_nearby_exhibits(&mut self, since_last_frame: Duration) {
        for g in self
            .level
            .floor_plan
            .valid_neighbors_at_dist(self.player_grid_pos(), 2)
        {
            if self.level.floor_plan.at(g) == GridElem::Exhibit {
                let object_idx = self.g2o.get(&g).unwrap();
                let object = &mut self.objects[object_idx.0];
                match object.kind {
                    ObjectKind::Exhibit { ref mut hiddenness } => {
                        *hiddenness =
                            (*hiddenness - since_last_frame.as_secs_f32() / 2.0).clamp(0., 1.);
                    }

                    _ => {
                        panic!(
                            "unexpectedly found {:?} at idx {:?}, grid idx {:?}",
                            object.kind, object_idx, g
                        );
                    }
                }
            }
        }
    }

    fn switch_view_mode(&mut self) {
        match self.vm {
            ViewMode::Topdown => {
                self.vm = ViewMode::Fps {
                    last_topdown_y: self.camera.mouse_y,
                };

                self.objects[self.light_object_idx].visible = false;
                self.audio.set_volume(1.0);
                self.camera.mouse_y = 0.0;
            }
            ViewMode::Fps { last_topdown_y } => {
                self.vm = ViewMode::Topdown;

                self.objects[self.light_object_idx].visible = true;
                self.audio.set_volume(0.5);
                self.camera.mouse_y = last_topdown_y;
            }
        }
    }

    fn handle_keys(&mut self, since_last_frame: Duration) -> ResolutionChangeAction {
        let mut answer = ResolutionChangeAction::DoNothing;

        if self.is_key_down(KeyCode::KeyW) {
            self.move_(Direction::Forward, since_last_frame);
        }

        if self.is_key_down(KeyCode::KeyS) {
            self.move_(Direction::Back, since_last_frame);
        }

        if self.is_key_down(KeyCode::KeyA) {
            self.move_(Direction::Left, since_last_frame);
        }

        if self.is_key_down(KeyCode::KeyD) {
            self.move_(Direction::Right, since_last_frame);
        }

        if self.was_key_pressed(KeyCode::Digit1) {
            self.renderer.render_settings.wireframe =
                self.renderer.render_settings.wireframe.next();
        }

        if self.is_key_down(KeyCode::KeyF) {
            match self.vm {
                ViewMode::Topdown => self.switch_view_mode(),
                ViewMode::Fps { .. } => {
                    // do nothing, we're already correct
                }
            }
            self.examine_nearby_exhibits(since_last_frame);
        } else {
            if self.vm != ViewMode::Topdown {
                self.switch_view_mode()
            }
        }
        if self.was_key_pressed(KeyCode::Digit4) {
            self.settings.rotate_objects = !self.settings.rotate_objects;

            if self.settings.rotate_objects {
                self.audio.set_track(audio::Track::Bear);
            } else {
                self.audio.set_track(audio::Track::Xpansive)
            }
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

        self.first_pressed_this_frame.clear();
        answer
    }

    fn update_light(&mut self) {
        match self.vm {
            ViewMode::Topdown => {
                self.light.dir = Mat3::from_rotation_y(self.camera.mouse_x) * vec3(0., 0., -1.);
            }
            ViewMode::Fps { last_topdown_y: _ } => {
                self.light.dir = self.camera.dir;
            }
        }

        // sway it gently
        let t = self.time_since_start.as_secs_f32();
        self.light.pos = self.player.pos.with_y(-3.6)
            + vec3(0.1 * f32::sin(t), -0.3 + 0.1 * f32::cos(0.7 * t), -0.1);
        self.light.dir += vec3(
            0.05 * f32::sin(0.8 * t),
            0.04 * f32::cos(1.1 * t),
            -0.06 * f32::sin(1.8 * t),
        );

        self.objects[self.light_object_idx].pos = self.light.pos;
    }

    fn animate_objects(&mut self, since_last_frame: Duration) {
        for object in self.objects.iter_mut() {
            match object.kind {
                ObjectKind::Exhibit { hiddenness } => {
                    if hiddenness <= 0. {
                        object.angle_y += since_last_frame.as_secs_f32();
                    }
                }
                ObjectKind::Light | ObjectKind::WallOrFloor | ObjectKind::Guard => {
                    // do nothing
                }
            }
        }
    }

    fn y_angle_clamp(vm: ViewMode) -> (f32, f32) {
        match vm {
            ViewMode::Topdown => (-f32::consts::PI / 2.1, -f32::consts::PI / 5.),
            ViewMode::Fps { .. } => (-f32::consts::PI / 2.1, f32::consts::PI / 3.),
        }
    }

    fn update_camera(&mut self) {
        let (y1, y2) = Self::y_angle_clamp(self.vm);
        self.camera.mouse_y = self.camera.mouse_y.clamp(y1, y2);

        self.camera.dir = Mat3::from_rotation_y(self.camera.mouse_x)
            * Mat3::from_rotation_x(self.camera.mouse_y)
            * vec3(0., 0., -1.);

        self.camera.pos = self.player.pos;

        match self.vm {
            ViewMode::Topdown => {
                self.camera.pos.y = 7.;
            }
            ViewMode::Fps { .. } => {
                self.camera.pos.y = -3.;
            }
        }
    }

    fn update_guards(&mut self, since_last_frame: Duration) {
        let alarm_enter_dist = 8.0;
        let alarm_exit_dist = 12.;
        for guard in self.guards.iter_mut() {
            let guard_speed = 3.;
            let guard_obj = &mut self.objects[guard.idx.0];

            let distance = (self.player.pos - guard_obj.pos).length();

            match guard.state {
                GuardState::Beat { facing: _ } if distance < alarm_enter_dist => {
                    guard.state = GuardState::Alarmed
                }

                GuardState::Alarmed if distance >= alarm_exit_dist => {
                    guard.state = GuardState::beat_facing_random(&mut self.rng)
                }

                _ => {
                    // do nothing
                }
            }

            match &mut guard.state {
                GuardState::Beat { facing } => {
                    let motion_dir = facing.to_world_dir();
                    let desired_pos =
                        guard_obj.pos + motion_dir * guard_speed * since_last_frame.as_secs_f32();

                    let desired_grid_pos = self.level.world2grid(desired_pos);
                    if self.level.floor_plan.at(desired_grid_pos) != GridElem::Empty {
                        *facing = facing.flip();
                    } else {
                        guard_obj.pos = desired_pos;
                    }

                    if Rng::random_range(&mut self.rng, 0..300) == 0 {
                        *facing = GridDir::random(&mut self.rng);
                    }

                    guard_obj.angle_y = facing.to_world_angle();
                }

                GuardState::Alarmed => {
                    let motion_dir = (self.player.pos - guard_obj.pos).with_y(0.0).normalize();

                    let desired_pos =
                        guard_obj.pos + motion_dir * guard_speed * since_last_frame.as_secs_f32();

                    guard_obj.pos = desired_pos;
                    let d = motion_dir.dot(vec3(0., 0., 1.));
                    let dor = motion_dir.x;
                    guard_obj.angle_y = dor.signum() * d.acos();
                }
            }

            self.spotlights[guard.spotlight.0].pos = guard_obj.pos;
        }
    }

    pub fn update(
        &mut self,
        since_last_frame: Duration,
        since_start: Duration,
    ) -> ResolutionChangeAction {
        self.time_since_start = since_start;

        let answer = self.handle_keys(since_last_frame);
        self.update_camera();
        self.animate_objects(since_last_frame);
        self.update_light();
        self.update_guards(since_last_frame);

        answer
    }

    fn draw_debug_grid_lines(&mut self) {
        let f = |v: Vec2| Vec3::new(v.x, -7., v.y);
        for neighbor in self
            .level
            .floor_plan
            .valid_neighbors_no_diagonals(self.level.world2grid(self.player.pos))
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

            self.renderer
                .debug_draw_line_in_world_space(self.player.pos, f(p1), BLACK);
        }

        let a = self.player.pos.with_y(-7.);
        let p = self.level.world2grid(self.player.pos);
        let (x, y) = self.level.floor_plan.to_xy(p);
        let b = self.level.grid2world(x, y).with_y(-7.);
        self.renderer.debug_draw_line_in_world_space(a, b, BLACK);
    }

    pub fn draw(&mut self, frame: &mut [u8]) -> RenderingResult {
        if self.settings.draw_debug_lines {
            self.draw_debug_grid_lines();
        }

        self.renderer.draw(
            &self.light,
            &self.spotlights,
            &self.camera,
            &self.objects,
            frame,
        )
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
        assert_eq!(g.first_empty(), g.from_xy(1, 1));
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
