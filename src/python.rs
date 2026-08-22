use crate::games::th05_c::GameState;
use crate::games::th05_c::watcher::TH05MemoryWatcher;
use crate::observation::schema1::ObservationBuilder;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use serde::{Deserialize, Serialize};
use std::io::ErrorKind;

/*
    Rust-Python bindings for Touhou PC-98 RL.
    Copyright (C) 2026  T. Liu and contributors

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

/// It is a snapshort which has not been passed to the map generator, so it is quite
/// small and is good for sitting in ram. When needed, pass it back to map.
#[pyclass(module = "pc98rl._native", from_py_object)]
#[derive(Clone, Serialize, Deserialize)]
pub struct RawFrame {
    pub(crate) state: GameState,
}

const REGULAR_BULLET_KILLBOX_HALF_EXTENT_PX: f32 = 4.0;
const REGULAR_BULLET_GRAZE_LEFT_PX: f32 = 16.0;
const REGULAR_BULLET_GRAZE_RIGHT_PX: f32 = 20.0;
const REGULAR_BULLET_GRAZE_VERTICAL_PX: f32 = 22.0;

fn playchar_speeds_subpixel(playchar: u8) -> (f32, f32) {
    match playchar {
        0 => (56.0, 40.0),
        1 => (64.0, 48.0),
        2 => (72.0, 52.0),
        3 => (56.0, 40.0),
        _ => (56.0, 40.0),
    }
}

fn movement_velocity_px(action: usize, playchar: u8) -> (f32, f32) {
    let movement = action % 9;
    let focused = action >= 9;
    let (mut aligned, mut diagonal) = playchar_speeds_subpixel(playchar);
    if focused {
        aligned = (aligned as i16 / 2) as f32;
        diagonal = (diagonal as i16 / 2) as f32;
    }
    aligned /= 16.0;
    diagonal /= 16.0;
    match movement {
        0 => (0.0, 0.0),
        1 => (-aligned, 0.0),
        2 => (aligned, 0.0),
        3 => (0.0, -aligned),
        4 => (0.0, aligned),
        5 => (-diagonal, -diagonal),
        6 => (-diagonal, diagonal),
        7 => (diagonal, -diagonal),
        8 => (diagonal, diagonal),
        _ => unreachable!(),
    }
}

fn regular_bullet_action_survival_frames(
    state: &GameState,
    horizon_frames: u8,
    extra_margin_px: f32,
) -> Vec<u8> {
    let mut survival_frames = vec![horizon_frames; 19];
    if state.player.invincibility_time > 0
        || state.player.invincible_via_bomb
        || state.player.miss_frame > 0
    {
        return survival_frames;
    }
    let (player_x, player_y) = state.player.pos.to_pixels();
    let extent = REGULAR_BULLET_KILLBOX_HALF_EXTENT_PX + extra_margin_px;
    for (action, survival) in survival_frames[..18].iter_mut().enumerate() {
        let (player_vx, player_vy) = movement_velocity_px(action, state.resident.playchar);
        for bullet in state.get_active_bullets() {
            // ReC98: spawn flags 3 and above are the delay cloud, and move
            // flags 4 and above are decay. Neither state has a hitbox.
            if bullet.spawn_flag >= 3 || bullet.move_flag >= 4 {
                continue;
            }
            let (bullet_x, bullet_y) = bullet.get_pixel_pos();
            let (bullet_vx, bullet_vy) = bullet.pos.velocity_pixels();
            // BSF_ACTIVE=2 becomes BSF_GRAZEABLE=0 before collision testing.
            // A grazeable bullet must enter the asymmetric graze box in one
            // frame before its 8x8 killbox becomes active in a later frame.
            let mut collision_active = bullet.spawn_flag == 1;
            for frame in 1..=horizon_frames {
                let time = frame as f32;
                let dx = bullet_x - player_x + (bullet_vx - player_vx) * time;
                let dy = bullet_y - player_y + (bullet_vy - player_vy) * time;
                if collision_active && dx.abs() <= extent && dy.abs() <= extent {
                    *survival = (*survival).min(frame - 1);
                    break;
                }
                if !collision_active
                    && dx >= -REGULAR_BULLET_GRAZE_LEFT_PX
                    && dx <= REGULAR_BULLET_GRAZE_RIGHT_PX
                    && dy.abs() <= REGULAR_BULLET_GRAZE_VERTICAL_PX
                {
                    collision_active = true;
                }
            }
        }
    }
    survival_frames
}

fn regular_bullet_action_mask(
    state: &GameState,
    horizon_frames: u8,
    extra_margin_px: f32,
) -> Vec<bool> {
    regular_bullet_action_survival_frames(state, horizon_frames, extra_margin_px)
        .into_iter()
        .map(|frames| frames == horizon_frames)
        .collect()
}
#[allow(clippy::new_without_default)]
// You clippy self touch the code here and got beated...
#[pymethods]
impl RawFrame {
    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let data = serde_json::to_vec(&self.state)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(PyBytes::new(py, &data))
    }

    pub fn __setstate__(&mut self, state: &Bound<'_, PyBytes>) -> PyResult<()> {
        let data = state.as_bytes();
        self.state = serde_json::from_slice(data)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(())
    }

    #[new]
    pub fn new() -> Self {
        Self {
            state: GameState {
                // [copy]

                // reader.rs
                resident: crate::games::th05_c::types::ResidentState {
                    rem_lives: 0,
                    credit_lives: 0,
                    rem_bombs: 0,
                    credit_bombs: 0,
                    stage: 0,
                    rank: 0,
                    playchar: 0,
                    stage_ascii: 0,
                    rand: 0,
                    bgm_mode: 0,
                    se_mode: 0,
                    shottype: 0,
                    debug_mode: 0,
                    end_type_ascii: 0,
                    end_sequence: 0,
                    score: 0,
                    graze: 0,
                    miss_count: 0,
                    bombs_used: 0,
                    items_spawned: 0,
                    items_collected: 0,
                    point_items_collected: 0,
                    max_valued_point_items_collected: 0,
                    enemies_killed: 0,
                    enemies_gone: 0,
                    frames: 0,
                    slow_frames: 0,
                    std_frames: 0,
                    demo_stage: 0,
                    demo_num: 0,
                    zunsoft_shown: 0,
                    turbo_mode: 0,
                    game_end_flag: 0,
                },
                player: crate::games::th05_c::types::PlayerState {
                    pos: crate::games::th05_c::types::PlayfieldMotion {
                        cur_x: 0,
                        cur_y: 0,
                        prev_x: 0,
                        prev_y: 0,
                        vel_x: 0,
                        vel_y: 0,
                    },
                    power: 0,
                    invincibility_time: 0,
                    control_lock: 0,
                    invincible_via_bomb: false,
                    player_is_hit: false,
                    miss_frame: 0,
                },
                bullets: vec![],
                enemies: vec![],
                items: vec![],
                boss: None,
                boss_2: None,
                midboss: None,
                lasers: vec![],
                cheeto_trails: vec![],
                custom_entities: vec![],
                firewaves: vec![],
                stage_collection: Default::default(),
                rem_bombs_internal: 0,
            },
        }
    }

    pub fn __repr__(&self) -> String {
        format!(
            "RawFrame(stage={}, frame={})",
            self.state.resident.stage, self.state.resident.frames
        )
    }

    /// Mask movement actions whose short projected path intersects an active
    /// regular-bullet killbox. Special projectiles are intentionally excluded.
    pub fn regular_bullet_action_mask(
        &self,
        horizon_frames: u8,
        extra_margin_px: f32,
    ) -> PyResult<Vec<bool>> {
        if horizon_frames == 0 || horizon_frames > 16 {
            return Err(PyValueError::new_err(
                "horizon_frames must be between 1 and 16",
            ));
        }
        if !extra_margin_px.is_finite() || extra_margin_px < 0.0 {
            return Err(PyValueError::new_err(
                "extra_margin_px must be finite and non-negative",
            ));
        }
        Ok(regular_bullet_action_mask(
            &self.state,
            horizon_frames,
            extra_margin_px,
        ))
    }

    /// Number of projected collision-free frames for every action.  This is
    /// used only to choose the least-immediate risk when no movement is safe
    /// for the complete requested horizon.
    pub fn regular_bullet_action_survival_frames(
        &self,
        horizon_frames: u8,
        extra_margin_px: f32,
    ) -> PyResult<Vec<u8>> {
        if horizon_frames == 0 || horizon_frames > 16 {
            return Err(PyValueError::new_err(
                "horizon_frames must be between 1 and 16",
            ));
        }
        if !extra_margin_px.is_finite() || extra_margin_px < 0.0 {
            return Err(PyValueError::new_err(
                "extra_margin_px must be finite and non-negative",
            ));
        }
        Ok(regular_bullet_action_survival_frames(
            &self.state,
            horizon_frames,
            extra_margin_px,
        ))
    }

    /// True only while TH05 can still cancel a registered collision with a
    /// bomb: player_is_hit before registration, or miss_time 40 down to 33.
    pub fn deathbomb_window_active(&self) -> bool {
        !self.state.player.invincible_via_bomb
            && (self.state.player.player_is_hit
                || (33..=40).contains(&self.state.player.miss_frame))
    }

    /// Read-only deployment diagnostics. These values are not policy features.
    pub fn stage_frame(&self) -> u32 {
        self.state.resident.frames
    }

    pub fn boss_phase(&self) -> Option<u8> {
        self.state.boss.as_ref().map(|boss| boss.phase)
    }

    pub fn boss_phase_frame(&self) -> Option<i16> {
        self.state.boss.as_ref().map(|boss| boss.phase_frame)
    }

    pub fn boss_hp(&self) -> Option<i16> {
        self.state.boss.as_ref().map(|boss| boss.hp)
    }
}

#[pyclass]
pub struct MemoryWatcher {
    pid: i32,
    inner: TH05MemoryWatcher,
    child: Option<std::process::Child>,
}

impl Drop for MemoryWatcher {
    fn drop(&mut self) {
        if let Some(ref mut child) = self.child {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

#[pymethods]
impl MemoryWatcher {
    /// If spawn_dosbox is True, a dosboxx process is spawned as a
    /// child and attached to main directly. So I would not require perfect timing
    /// to launch the agent.
    #[new]
    #[pyo3(signature = (spawn_dosbox=false, pid=None, image_path=None, dosbox_executable=None))]
    pub fn new(
        spawn_dosbox: bool,
        pid: Option<i32>,
        image_path: Option<String>,
        dosbox_executable: Option<String>,
    ) -> PyResult<Self> {
        if spawn_dosbox {
            if pid.is_some() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "pid cannot be combined with spawn_dosbox=True",
                ));
            }
            Self::new_spawn(image_path.as_deref(), dosbox_executable.as_deref())
        } else {
            if image_path.is_some() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "image_path requires spawn_dosbox=True",
                ));
            }
            if dosbox_executable.is_some() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "dosbox_executable requires spawn_dosbox=True",
                ));
            }
            Self::new_attach(pid)
        }
    }

    pub fn pause_game(&self) {
        let _ = nix::sys::signal::kill(
            nix::unistd::Pid::from_raw(self.pid),
            nix::sys::signal::Signal::SIGSTOP,
        );
    }

    pub fn resume_game(&self) {
        let _ = nix::sys::signal::kill(
            nix::unistd::Pid::from_raw(self.pid),
            nix::sys::signal::Signal::SIGCONT,
        );
    }

    pub fn release_action(&mut self) -> PyResult<()> {
        self.inner
            .release_action()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    pub fn apply_action(&mut self, action: usize) -> PyResult<()> {
        self.inner
            .apply_action(action)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    pub fn apply_action_guarded(
        &mut self,
        action: usize,
        deathbomb_guard: bool,
    ) -> PyResult<bool> {
        self.inner
            .apply_action_guarded(action, deathbomb_guard)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Read the current game state and return MO rewards.
    /// if you want more documents please read docs dir.
    pub fn read_state(&mut self) -> PyResult<Option<(Vec<f32>, Vec<f32>, u8, Vec<f32>, RawFrame)>> {
        let prev_state = self.inner.last_state.clone();

        let state = match self.inner.try_read_state() {
            Some(s) => s,
            None => return Ok(None),
        };

        let (features, maps) = self.inner.observation_comp(&state);
        let game_end_flag = state.resident.game_end_flag;

        let rewards = if let Some(ref prev) = prev_state {
            let mut r = crate::observation::schema1::reward::calculate_reward_m(Some(prev), &state);
            // Apply episode-end good things
            if game_end_flag == 1 {
                r[0] -= 100.0;
            } else if game_end_flag == 2 {
                r[0] += 100.0;
            }
            r
        } else {
            vec![0.0, 0.0, 0.0]
        };

        let raw_frame = RawFrame { state };
        Ok(Some((features, maps, game_end_flag, rewards, raw_frame)))
    }

    /// Read the same game state without constructing dense spatial maps.
    ///
    /// The compact 273-float observation already contains player/boss state,
    /// the nearest 16 regular bullets, the nearest 16 special projectiles, and
    /// drop features.  Returning it directly cuts per-step observation traffic
    /// from roughly 0.81 MiB to 1.1 KiB.
    pub fn read_features(&mut self) -> PyResult<Option<(Vec<f32>, u8, Vec<f32>, RawFrame)>> {
        let prev_state = self.inner.last_state.clone();

        let state = match self.inner.try_read_state() {
            Some(s) => s,
            None => return Ok(None),
        };

        let features = self.inner.observation_features(&state);
        let game_end_flag = state.resident.game_end_flag;
        let rewards = if let Some(ref prev) = prev_state {
            let mut rewards =
                crate::observation::schema1::reward::calculate_reward_m(Some(prev), &state);
            if game_end_flag == 1 {
                rewards[0] -= 100.0;
            } else if game_end_flag == 2 {
                rewards[0] += 100.0;
            }
            rewards
        } else {
            vec![0.0, 0.0, 0.0]
        };

        Ok(Some((features, game_end_flag, rewards, RawFrame { state })))
    }

    pub fn clear_state(&mut self) {
        self.inner.last_state = None;
    }

    /// Read the keys pressed by me.
    pub fn read_human_action(&mut self) -> PyResult<usize> {
        let Some(ref mut device) = self.inner.keyboard_device else {
            return Err(PyRuntimeError::new_err(
                "No keyboard device found. Is your kernel outdated? How you started Linux?",
            ));
        };
        // non blocking for the less latency cuz I won't hold bomb key all the time like it.
        match device.fetch_events() {
            Ok(events) => {
                for _event in events {
                    // thanks for the optimization!
                }
            }
            Err(err) if err.kind() == ErrorKind::WouldBlock => {
                // No new keyboard events this frame.
            }
            Err(err) => {
                return Err(PyRuntimeError::new_err(format!(
                    "Failed to fetch keyboard events: {err}"
                )));
            }
        }

        let state = device.cached_state();

        let is_pressed = |key: evdev::KeyCode| -> bool {
            state.key_vals().is_some_and(|keys| keys.contains(key))
        };

        let up = is_pressed(evdev::KeyCode::KEY_UP);
        let down = is_pressed(evdev::KeyCode::KEY_DOWN);
        let left = is_pressed(evdev::KeyCode::KEY_LEFT);
        let right = is_pressed(evdev::KeyCode::KEY_RIGHT);
        let bomb = is_pressed(evdev::KeyCode::KEY_X);
        let shift =
            is_pressed(evdev::KeyCode::KEY_LEFTSHIFT) || is_pressed(evdev::KeyCode::KEY_RIGHTSHIFT);

        if bomb {
            return Ok(18); // bomb idx is 18
        }

        let move_left = left && !right;
        let move_right = right && !left;
        let move_up = up && !down;
        let move_down = down && !up;

        let movement = if move_left && move_up {
            crate::games::th05_c::key::MoveAction::LeftUp
        } else if move_left && move_down {
            crate::games::th05_c::key::MoveAction::LeftDown
        } else if move_right && move_up {
            crate::games::th05_c::key::MoveAction::RightUp
        } else if move_right && move_down {
            crate::games::th05_c::key::MoveAction::RightDown
        } else if move_left {
            crate::games::th05_c::key::MoveAction::Left
        } else if move_right {
            crate::games::th05_c::key::MoveAction::Right
        } else if move_up {
            crate::games::th05_c::key::MoveAction::Up
        } else if move_down {
            crate::games::th05_c::key::MoveAction::Down
        } else {
            crate::games::th05_c::key::MoveAction::Idle
        };

        let action = crate::games::th05_c::key::DiscreteAction::Move { movement, shift };
        Ok(action.to_index())
    }

    /// Terminate (or, SIGKILL actually) the managed DOSBox-X child process, if any.
    pub fn terminate(&mut self) {
        if let Some(ref mut child) = self.child {
            let _ = child.kill();
            let _ = child.wait();
        }
        self.child = None;
    }
    pub fn pid(&self) -> i32 {
        self.pid
    }
}

impl MemoryWatcher {
    fn env_is_nonempty(name: &str) -> bool {
        std::env::var_os(name).is_some_and(|value| !value.is_empty())
    }

    fn should_force_dummy_sdl_for(
        has_video_driver: bool,
        has_x11_display: bool,
        has_wayland_display: bool,
    ) -> bool {
        !has_video_driver && !has_x11_display && !has_wayland_display
    }

    fn should_force_dummy_sdl() -> bool {
        Self::should_force_dummy_sdl_for(
            Self::env_is_nonempty("SDL_VIDEODRIVER"),
            Self::env_is_nonempty("DISPLAY"),
            Self::env_is_nonempty("WAYLAND_DISPLAY"),
        )
    }

    fn cfg_keydev(inner: &mut TH05MemoryWatcher) -> PyResult<()> {
        let Some(device) = inner.keyboard_device.as_mut() else {
            return Ok(());
        };

        device.set_nonblocking(true).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to set keyboard device nonblocking: {e}"))
        })?;

        // there are no pending keyboard events
        match device.fetch_events() {
            Ok(events) => {
                for _event in events {
                    // Again thanks
                }
            }
            Err(err) if err.kind() == ErrorKind::WouldBlock => {}
            Err(err) => {
                return Err(PyRuntimeError::new_err(format!(
                    "Failed to fetch initial keyboard events: {err}"
                )));
            }
        }

        Ok(())
    }

    /// Attach to one, very old.
    fn new_attach(requested_pid: Option<i32>) -> PyResult<Self> {
        let pid = match requested_pid {
            Some(pid) => pid,
            None => crate::memory::find_pid()
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?,
        };
        let mut inner = TH05MemoryWatcher::new(pid)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        inner
            .initialize()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        Ok(Self {
            pid,
            inner,
            child: None,
        })
    }

    fn new_spawn(image_path: Option<&str>, dosbox_executable: Option<&str>) -> PyResult<Self> {
        use std::process::Command;

        let project_dir = std::env::current_dir()
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to get cwd: {}", e))
            })?;
        let export_dir = project_dir.join("export");

        let executable = dosbox_executable.unwrap_or("dosbox-x");
        let mut command = Command::new(executable);
        // CPU training commonly runs from a non-interactive shell.  SDL exits
        // immediately when neither X11 nor Wayland is available, but its dummy
        // video backend is sufficient because observations come from guest RAM.
        if Self::should_force_dummy_sdl() {
            command.env("SDL_VIDEODRIVER", "dummy");
        }
        if let Some(image_path) = image_path {
            let image_path = std::fs::canonicalize(image_path).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "Failed to resolve image_path '{}': {}",
                    image_path, e
                ))
            })?;
            let image_path = image_path.to_string_lossy();
            let conf_path = export_dir.join("default.conf");
            command
                .args(["-conf", &conf_path.to_string_lossy()])
                .args(["-c", &format!("imgmount c \"{}\" -partidx 0", image_path)])
                .args(["-c", "c:", "-c", "cd kaiki", "-c", "game", "-fastlaunch"])
                .current_dir(&project_dir);
        } else {
            command
                .args([
                    ".",
                    "-conf",
                    "./default.conf",
                    "-c",
                    "mount c .",
                    "-c",
                    "c:",
                    "-c",
                    "game",
                    "-fastlaunch",
                ])
                .current_dir(&export_dir);
        }

        let mut child = command
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to spawn DOSBox-X from '{}': {}",
                    executable, e
                ))
            })?;

        let pid = child.id() as i32;
        tracing::info!("Spawned DOSBox-X child process with PID {}", pid);

        // Let it start and wait for bios and game render.
        std::thread::sleep(std::time::Duration::from_millis(500));

        // cuz sometimes it is slow so retry.
        let max_retries = 30;
        let mut last_err = String::new();
        for attempt in 1..=max_retries {
            if let Some(status) = child.try_wait().map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to query DOSBox-X child '{}': {}",
                    executable, e
                ))
            })? {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "DOSBox-X (PID {}) exited before memory attach with {}. \
                     Check the emulator configuration and SDL video environment.",
                    pid, status
                )));
            }
            match TH05MemoryWatcher::new(pid) {
                Ok(mut inner) => match inner.initialize() {
                    Ok(()) => {
                        Self::cfg_keydev(&mut inner)?;

                        tracing::info!(
                            "MemoryWatcher attached to PID {} after {} attempt(s)",
                            pid,
                            attempt
                        );

                        return Ok(Self {
                            pid,
                            inner,
                            child: Some(child),
                        });
                    }
                    Err(e) => {
                        last_err = e.to_string();
                    }
                },
                Err(e) => {
                    last_err = e.to_string();
                }
            }
            if attempt < max_retries {
                std::thread::sleep(std::time::Duration::from_millis(500));
            }
        }

        // Failed after all retries.
        let _ = child.kill();
        let _ = child.wait();
        Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "DOSBox-X (PID {}) started but game structures not found after {}: {}. \
                Are child running? Have you built dosbox-x from source?",
            pid, max_retries, last_err
        )))
    }
}

/// Resolve a curriculum event from jason. `program` is the cc from Python (i cannot name it
/// better)
/// btw, specific cfg gen is not a good name for that, really.
#[pyfunction]
#[pyo3(signature = (program, event, current_json=None))]
pub fn cfg_execute(program: &str, event: &str, current_json: Option<&str>) -> String {
    crate::cfg::execute_cc_json(program, event, current_json)
}

/// Write a json coded directly
#[pyfunction]
pub fn cfg_write_json(path: &str, cfg_json: &str) {
    crate::cfg::write_cfg_json(std::path::Path::new(path), cfg_json)
}

#[pyfunction]
pub fn init_logging() {
    crate::logging::init_tracing();
}

/// Reconstruct spatial maps from before the raw frames, direct game raw frame.
/// So why is it not used? Well, in 1ms it has already done. So no need.
/// (Unless you have serious memory lackages)
#[pyfunction]
pub fn rec_maps_bytes<'py>(
    py: Python<'py>,
    frames: Vec<PyRef<'_, RawFrame>>,
) -> PyResult<Bound<'py, PyBytes>> {
    let builder = ObservationBuilder::default();
    let num_frames = frames.len();
    let size = builder.grid_w * builder.grid_h * 24; // 211968 ?
    let mut buffer = vec![0.0f32; num_frames * size];

    for (i, f) in frames.iter().enumerate() {
        let obs = builder.build_observation(&f.state);
        let tensor = obs.to_map_tensor();
        let start = i * size;
        buffer[start..start + size].copy_from_slice(&tensor);
    }

    let byte_slice = unsafe {
        std::slice::from_raw_parts(
            buffer.as_ptr() as *const u8,
            buffer.len() * std::mem::size_of::<f32>(),
        )
    };

    Ok(PyBytes::new(py, byte_slice))
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<MemoryWatcher>()?;
    m.add_class::<RawFrame>()?;
    m.add_function(wrap_pyfunction!(cfg_execute, m)?)?;
    m.add_function(wrap_pyfunction!(cfg_write_json, m)?)?;
    m.add_function(wrap_pyfunction!(init_logging, m)?)?;
    m.add_function(wrap_pyfunction!(rec_maps_bytes, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        MemoryWatcher, RawFrame, regular_bullet_action_mask,
        regular_bullet_action_survival_frames,
    };
    use crate::games::th05_c::types::{Bullet, PlayfieldMotion};

    #[test]
    fn dummy_video_is_only_for_displayless_launches() {
        assert!(MemoryWatcher::should_force_dummy_sdl_for(
            false, false, false
        ));
        assert!(!MemoryWatcher::should_force_dummy_sdl_for(
            true, false, false
        ));
        assert!(!MemoryWatcher::should_force_dummy_sdl_for(
            false, true, false
        ));
        assert!(!MemoryWatcher::should_force_dummy_sdl_for(
            false, false, true
        ));
    }

    #[test]
    fn audited_bullet_mask_keeps_escape_actions() {
        let mut frame = RawFrame::new();
        frame.state.resident.playchar = 2;
        frame.state.player.pos.cur_x = 100 * 16;
        frame.state.player.pos.cur_y = 100 * 16;
        frame.state.bullets.push(Bullet {
            flag: 1,
            age: 10,
            pos: PlayfieldMotion {
                cur_x: 112 * 16,
                cur_y: 100 * 16,
                prev_x: 116 * 16,
                prev_y: 100 * 16,
                vel_x: -4 * 16,
                vel_y: 0,
            },
            from_group: 0,
            speed_cur: 0,
            angle: 0,
            spawn_flag: 1,
            move_flag: 2,
            special_motion: 0,
            speed_final: 0,
            decel_time_or_turns: 0,
            decel_delta_or_angle: 0,
            patnum: 0,
        });

        let mask = regular_bullet_action_mask(&frame.state, 2, 0.0);
        assert!(!mask[0]);
        assert!(mask[1]);
        assert!(!mask[2]);
        assert!(mask[18]);
        let survival = regular_bullet_action_survival_frames(&frame.state, 2, 0.0);
        assert!(survival[0] < 2);
        assert_eq!(survival[1], 2);
        assert!(survival[2] < 2);
        assert_eq!(survival[18], 2);
    }

    #[test]
    fn cloud_or_invincible_bullets_do_not_constrain_actions() {
        let mut frame = RawFrame::new();
        frame.state.player.pos.cur_x = 100 * 16;
        frame.state.player.pos.cur_y = 100 * 16;
        frame.state.bullets.push(Bullet {
            flag: 1,
            age: 0,
            pos: PlayfieldMotion {
                cur_x: 100 * 16,
                cur_y: 100 * 16,
                prev_x: 100 * 16,
                prev_y: 100 * 16,
                vel_x: 0,
                vel_y: 0,
            },
            from_group: 0,
            speed_cur: 0,
            angle: 0,
            spawn_flag: 3,
            move_flag: 2,
            special_motion: 0,
            speed_final: 0,
            decel_time_or_turns: 0,
            decel_delta_or_angle: 0,
            patnum: 0,
        });
        assert!(
            regular_bullet_action_mask(&frame.state, 2, 0.0)
                .iter()
                .all(|v| *v)
        );

        frame.state.bullets[0].spawn_flag = 1;
        frame.state.player.invincibility_time = 1;
        assert!(
            regular_bullet_action_mask(&frame.state, 2, 0.0)
                .iter()
                .all(|v| *v)
        );
    }

    #[test]
    fn grazeable_bullet_only_collides_on_a_later_frame() {
        let mut frame = RawFrame::new();
        frame.state.resident.playchar = 2;
        frame.state.player.pos.cur_x = 100 * 16;
        frame.state.player.pos.cur_y = 100 * 16;
        frame.state.bullets.push(Bullet {
            flag: 1,
            age: 10,
            pos: PlayfieldMotion {
                cur_x: 112 * 16,
                cur_y: 100 * 16,
                prev_x: 116 * 16,
                prev_y: 100 * 16,
                vel_x: -4 * 16,
                vel_y: 0,
            },
            from_group: 0,
            speed_cur: 0,
            angle: 0,
            spawn_flag: 0,
            move_flag: 2,
            special_motion: 0,
            speed_final: 0,
            decel_time_or_turns: 0,
            decel_delta_or_angle: 0,
            patnum: 0,
        });

        assert!(regular_bullet_action_mask(&frame.state, 1, 0.0)[0]);
        assert!(!regular_bullet_action_mask(&frame.state, 2, 0.0)[0]);
        assert!(regular_bullet_action_mask(&frame.state, 2, 0.0)[1]);
    }

    #[test]
    fn deathbomb_window_matches_th05_miss_timer() {
        let mut frame = RawFrame::new();
        frame.state.player.miss_frame = 40;
        assert!(frame.deathbomb_window_active());
        frame.state.player.miss_frame = 33;
        assert!(frame.deathbomb_window_active());
        frame.state.player.miss_frame = 32;
        assert!(!frame.deathbomb_window_active());

        frame.state.player.player_is_hit = true;
        assert!(frame.deathbomb_window_active());
        frame.state.player.invincible_via_bomb = true;
        assert!(!frame.deathbomb_window_active());
    }

    #[test]
    fn empty_raw_frame_has_progress_diagnostics() {
        let frame = RawFrame::new();
        assert_eq!(frame.stage_frame(), 0);
        assert_eq!(frame.boss_phase(), None);
        assert_eq!(frame.boss_phase_frame(), None);
        assert_eq!(frame.boss_hp(), None);
    }
}
