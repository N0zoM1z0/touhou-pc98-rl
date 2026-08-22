"""Gymnasium environment for the CPU/HDI TH05 execution path."""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from . import _native

from .contracts import ActionDescriptor, ConstraintProvider, KinematicSpec
from .model import FEATURE_DIM


ACTION_DIM = 19
DEFAULT_REWARD_SCALES = np.asarray((0.01, 0.001, 0.01), dtype=np.float32)
DEFAULT_REWARD_WEIGHTS = np.asarray((1.0, 1.0, 0.25), dtype=np.float32)
TH05_KINEMATICS = KinematicSpec(
    position_scale=(384.0, 368.0),
    velocity_scale=(12.0, 12.0),
    horizon_steps=60.0,
)

_MOVEMENT_DESCRIPTORS = (
    (0, 0),
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
)
TH05_ACTIONS = tuple(
    ActionDescriptor(move_x=x, move_y=y, primary=primary)
    for primary in (False, True)
    for x, y in _MOVEMENT_DESCRIPTORS
) + (ActionDescriptor(bomb=True),)


class TH05Constraints(ConstraintProvider):
    """Exact availability and clamp-equivalence constraints for TH05 actions."""

    boundary_epsilon = 1e-7

    def valid_actions(self, observation: np.ndarray) -> np.ndarray:
        observation = np.asarray(observation, dtype=np.float32)
        if observation.shape[-1] != FEATURE_DIM:
            raise ValueError(f"expected {FEATURE_DIM} features")
        mask = np.ones((*observation.shape[:-1], ACTION_DIM), dtype=np.bool_)
        left_blocked = observation[..., 4] <= self.boundary_epsilon
        right_blocked = observation[..., 5] <= self.boundary_epsilon
        top_blocked = observation[..., 6] <= self.boundary_epsilon
        bottom_blocked = observation[..., 7] <= self.boundary_epsilon
        for action, descriptor in enumerate(TH05_ACTIONS[:-1]):
            blocked = np.zeros(observation.shape[:-1], dtype=np.bool_)
            if descriptor.move_x < 0:
                blocked |= left_blocked
            elif descriptor.move_x > 0:
                blocked |= right_blocked
            if descriptor.move_y < 0:
                blocked |= top_blocked
            elif descriptor.move_y > 0:
                blocked |= bottom_blocked
            mask[..., action] &= ~blocked

        # Bombs are the only truly resource-gated command in this adapter.
        mask[..., -1] = observation[..., 10] > self.boundary_epsilon
        return mask


TH05_CONSTRAINTS = TH05Constraints()


def describe_th05_scenario(observation: np.ndarray) -> dict[str, int | str]:
    """Decode stable resident configuration fields from a compact observation."""
    observation = np.asarray(observation, dtype=np.float32)
    if observation.shape != (FEATURE_DIM,):
        raise ValueError(f"expected {FEATURE_DIM} features")
    stage_index = int(np.rint(observation[13] * 6.0))
    return {
        "stage": "extra" if stage_index == 6 else stage_index + 1,
        "patch_stage_index": stage_index,
        "character": int(np.rint(observation[12] * 3.0)),
        "rank": int(np.rint(observation[16] * 3.0)),
        "initial_power": int(np.rint(observation[8] * 128.0)),
        "configured_lives": int(np.rint(observation[14] * 8.0)),
        "configured_bombs": int(np.rint(observation[15] * 8.0)),
    }


def resolve_dosbox_executable(candidate: str | Path | None = None) -> Path:
    """Resolve DOSBox-X without requiring callers to modify ``PATH``."""
    requested = candidate or os.environ.get("PC98RL_DOSBOX_X")
    if requested is not None:
        requested_path = Path(requested).expanduser()
        if requested_path.parent != Path("."):
            resolved = requested_path.resolve()
        else:
            found = shutil.which(str(requested_path))
            resolved = Path(found) if found else requested_path.resolve()
        if resolved.is_file() and os.access(resolved, os.X_OK):
            return resolved
        raise FileNotFoundError(f"DOSBox-X executable is not runnable: {resolved}")

    found = shutil.which("dosbox-x")
    if found:
        return Path(found).resolve()
    local = Path(__file__).resolve().parents[1] / "external/dosbox-x-install/bin/dosbox-x"
    if local.is_file() and os.access(local, os.X_OK):
        return local
    raise FileNotFoundError(
        "DOSBox-X was not found; build the local emulator or set PC98RL_DOSBOX_X"
    )


class TH05CPUEnv(gym.Env):
    """One isolated TH05 instance backed by a private copy of a patched HDI.

    Each reset restores the small (about 21 MiB) disk image.  This makes
    episodes deterministic with respect to game files and lets many rollout
    workers run without racing on score/config writes.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        image_template: str | Path,
        *,
        frame_interval_s: float = 0.036,
        reward_scales: np.ndarray = DEFAULT_REWARD_SCALES,
        reward_weights: np.ndarray = DEFAULT_REWARD_WEIGHTS,
        warmup_timeout_s: float = 5.0,
        spawn_retries: int = 3,
        dosbox_executable: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.image_template = Path(image_template).expanduser().resolve()
        if not self.image_template.is_file():
            raise FileNotFoundError(self.image_template)

        self.frame_interval_s = float(frame_interval_s)
        self.reward_scales = np.asarray(reward_scales, dtype=np.float32)
        self.reward_weights = np.asarray(reward_weights, dtype=np.float32)
        if self.reward_scales.shape != (3,) or self.reward_weights.shape != (3,):
            raise ValueError("reward_scales and reward_weights must have shape (3,)")

        self.warmup_timeout_s = float(warmup_timeout_s)
        self.dosbox_executable = resolve_dosbox_executable(dosbox_executable)
        self.spawn_retries = int(spawn_retries)
        if self.spawn_retries < 1:
            raise ValueError("spawn_retries must be positive")
        self.action_space = gym.spaces.Discrete(ACTION_DIM)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(FEATURE_DIM,), dtype=np.float32
        )

        self._tempdir = tempfile.TemporaryDirectory(prefix="th05-cpu-")
        self._working_image = Path(self._tempdir.name) / "game.hdi"
        self._watcher: _native.MemoryWatcher | None = None
        self._observation: np.ndarray | None = None
        self._next_deadline = 0.0
        self._closed = False

    def _stop(self) -> None:
        if self._watcher is not None:
            try:
                self._watcher.release_action()
            finally:
                self._watcher.terminate()
                self._watcher = None

    def _read_ready_state(self) -> tuple[np.ndarray, dict[str, Any]]:
        assert self._watcher is not None
        deadline = time.monotonic() + self.warmup_timeout_s
        last_state = None
        while time.monotonic() < deadline:
            state = self._watcher.read_features()
            if state is not None:
                features, end_flag, rewards, raw_frame = state
                observation = np.asarray(features, dtype=np.float32)
                if observation.shape == (FEATURE_DIM,):
                    last_state = (observation, end_flag, rewards, raw_frame)
                    # The resident exists before MAIN initializes the player,
                    # and a new stage then spends roughly three seconds in an
                    # invincible input lock.  Returning that prefix gives PPO
                    # survival reward for actions the game cannot execute.
                    if observation[1] > 0.01 and observation[11] < 0.5:
                        break
            time.sleep(0.01)

        if (
            last_state is None
            or last_state[0][1] <= 0.01
            or last_state[0][11] >= 0.5
        ):
            raise RuntimeError("TH05 started but did not reach a controllable gameplay state")
        observation, end_flag, rewards, raw_frame = last_state
        return observation, {
            "end_flag": int(end_flag),
            "reward_vector": np.asarray(rewards, dtype=np.float32),
            "raw_frame": raw_frame,
            "action_mask": TH05_CONSTRAINTS.valid_actions(observation),
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del options
        super().reset(seed=seed)
        if self._closed:
            raise RuntimeError("cannot reset a closed TH05CPUEnv")

        self._stop()
        last_error = None
        for attempt in range(self.spawn_retries):
            shutil.copyfile(self.image_template, self._working_image)
            try:
                self._watcher = _native.MemoryWatcher(
                    spawn_dosbox=True,
                    image_path=str(self._working_image),
                    dosbox_executable=str(self.dosbox_executable),
                )
                observation, info = self._read_ready_state()
                self._observation = observation
                self._next_deadline = time.monotonic()
                return observation.copy(), info
            except RuntimeError as error:
                last_error = error
                self._stop()
                if attempt + 1 < self.spawn_retries:
                    time.sleep(0.2 * (attempt + 1))
        raise RuntimeError(
            f"failed to spawn a readable TH05 process after {self.spawn_retries} attempts"
        ) from last_error

    def step(self, action: int):
        if self._watcher is None or self._observation is None:
            raise RuntimeError("reset() must be called before step()")
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action {action}")

        self._watcher.apply_action(int(action))
        self._next_deadline += self.frame_interval_s
        now = time.monotonic()
        if self._next_deadline > now:
            time.sleep(self._next_deadline - now)
        elif now - self._next_deadline > self.frame_interval_s:
            # Do not accumulate scheduler lag after a slow inference/update.
            self._next_deadline = now

        state = self._watcher.read_features()
        if state is None:
            raise RuntimeError("lost TH05 state while stepping")
        features, end_flag, rewards, raw_frame = state
        observation = np.asarray(features, dtype=np.float32)
        reward_vector = np.asarray(rewards, dtype=np.float32)
        scaled_reward_vector = reward_vector * self.reward_scales
        reward = float(np.dot(scaled_reward_vector, self.reward_weights))
        self._observation = observation

        terminated = int(end_flag) != 0
        info = {
            "end_flag": int(end_flag),
            "success": int(end_flag) == 2,
            "reward_vector": reward_vector,
            "scaled_reward_vector": scaled_reward_vector,
            "raw_frame": raw_frame,
            "action_mask": TH05_CONSTRAINTS.valid_actions(observation),
        }
        return observation.copy(), reward, terminated, False, info

    def close(self) -> None:
        if self._closed:
            return
        self._stop()
        self._tempdir.cleanup()
        self._closed = True


__all__ = [
    "ACTION_DIM",
    "TH05_ACTIONS",
    "TH05_CONSTRAINTS",
    "TH05_KINEMATICS",
    "TH05CPUEnv",
]
