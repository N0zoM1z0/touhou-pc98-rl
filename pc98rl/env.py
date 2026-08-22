"""Gymnasium environment for the CPU/HDI TH05 execution path."""

from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from . import _native

from .model import FEATURE_DIM


ACTION_DIM = 19
DEFAULT_REWARD_SCALES = np.asarray((0.01, 0.001, 0.01), dtype=np.float32)
DEFAULT_REWARD_WEIGHTS = np.asarray((1.0, 1.0, 0.25), dtype=np.float32)


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
                    # The resident exists slightly before MAIN initializes the
                    # player.  A positive Y coordinate distinguishes gameplay
                    # from that short loader window.
                    if observation[1] > 0.01:
                        break
            time.sleep(0.01)

        if last_state is None or last_state[0][1] <= 0.01:
            raise RuntimeError("TH05 started but did not reach an initialized gameplay state")
        observation, end_flag, rewards, raw_frame = last_state
        return observation, {
            "end_flag": int(end_flag),
            "reward_vector": np.asarray(rewards, dtype=np.float32),
            "raw_frame": raw_frame,
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
                    spawn_dosbox=True, image_path=str(self._working_image)
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
        }
        return observation.copy(), reward, terminated, False, info

    def close(self) -> None:
        if self._closed:
            return
        self._stop()
        self._tempdir.cleanup()
        self._closed = True


__all__ = ["ACTION_DIM", "TH05CPUEnv"]
