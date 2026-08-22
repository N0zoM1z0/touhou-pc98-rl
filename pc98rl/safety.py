"""Adapter-level last-resort safety interventions."""

from __future__ import annotations

import numpy as np

from .contracts import KinematicSpec
from .model import ENTITY_COUNT, ENTITY_DIM, FEATURE_DIM, GLOBAL_DIM


class EmergencyBombShield:
    """Force a bomb only for an imminent constant-velocity collision.

    This is deliberately narrower than a hand-written controller: it never
    chooses movement and stays inactive when the player is invincible or has no
    bombs.  Returning an action mask lets PPO account for the intervention in
    both rollout log-probabilities and optimization.
    """

    action_dim = 19
    bomb_action = 18

    def __init__(
        self,
        kinematics: KinematicSpec,
        *,
        clearance_px: float = 10.0,
        horizon_steps: float = 6.0,
    ) -> None:
        if clearance_px <= 0.0:
            raise ValueError("clearance_px must be positive")
        if horizon_steps <= 0.0:
            raise ValueError("horizon_steps must be positive")
        self.kinematics = kinematics
        self.clearance_px = float(clearance_px)
        self.horizon_steps = float(horizon_steps)

    def _minimum_clearance(self, observation: np.ndarray) -> float:
        projectile_start = GLOBAL_DIM
        bullet_start = projectile_start + ENTITY_COUNT * ENTITY_DIM
        drop_start = bullet_start + ENTITY_COUNT * ENTITY_DIM
        entities = np.concatenate(
            (
                observation[projectile_start:bullet_start].reshape(
                    ENTITY_COUNT, ENTITY_DIM
                ),
                observation[bullet_start:drop_start].reshape(
                    ENTITY_COUNT, ENTITY_DIM
                ),
            ),
            axis=0,
        )
        present = (np.abs(entities[:, :6]).sum(axis=1) > 1e-7) | (
            entities[:, 6] < 1.0 - 1e-7
        )
        entities = entities[present]
        if not len(entities):
            return float("inf")

        relative_position = entities[:, :2] * np.asarray(
            self.kinematics.position_scale, dtype=np.float32
        )
        player_velocity = observation[2:4] * np.asarray(
            self.kinematics.velocity_scale, dtype=np.float32
        )
        relative_velocity = (
            entities[:, 2:4]
            * np.asarray(self.kinematics.velocity_scale, dtype=np.float32)
            - player_velocity
        )
        speed_squared = np.square(relative_velocity).sum(axis=1)
        closest_time = np.zeros(len(entities), dtype=np.float32)
        moving = speed_squared > 1e-8
        closest_time[moving] = np.clip(
            -np.sum(relative_position[moving] * relative_velocity[moving], axis=1)
            / speed_squared[moving],
            0.0,
            self.horizon_steps,
        )
        closest = relative_position + relative_velocity * closest_time[:, None]
        return float(np.linalg.norm(closest, axis=1).min())

    def should_intervene(self, observation: np.ndarray) -> bool:
        observation = np.asarray(observation, dtype=np.float32)
        if observation.shape != (FEATURE_DIM,):
            raise ValueError(f"expected {FEATURE_DIM} features")
        bombs_available = observation[10] > 1e-7
        vulnerable = observation[11] < 0.5
        return bool(
            bombs_available
            and vulnerable
            and self._minimum_clearance(observation) <= self.clearance_px
        )

    def apply(
        self, observation: np.ndarray, base_mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray | np.bool_]:
        """Return the effective mask and per-observation intervention flag."""
        observation = np.asarray(observation, dtype=np.float32)
        base_mask = np.asarray(base_mask, dtype=np.bool_)
        if observation.shape[-1] != FEATURE_DIM:
            raise ValueError(f"expected {FEATURE_DIM} features")
        if base_mask.shape != (*observation.shape[:-1], self.action_dim):
            raise ValueError("base mask shape does not match observations")

        flat_observation = observation.reshape(-1, FEATURE_DIM)
        flat_mask = base_mask.reshape(-1, self.action_dim).copy()
        interventions = np.asarray(
            [self.should_intervene(item) for item in flat_observation], dtype=np.bool_
        )
        for index in np.flatnonzero(interventions):
            if flat_mask[index, self.bomb_action]:
                flat_mask[index] = False
                flat_mask[index, self.bomb_action] = True
            else:
                interventions[index] = False
        mask = flat_mask.reshape(base_mask.shape)
        flags = interventions.reshape(observation.shape[:-1])
        if observation.ndim == 1:
            return mask, flags[()]
        return mask, flags


class AuditedRegularBulletShield:
    """Combine the native ReC98-derived regular-bullet mask with base rules."""

    def __init__(
        self,
        *,
        horizon_frames: int = 2,
        extra_margin_px: float = 0.0,
        least_risk_fallback: bool = False,
    ):
        if not 1 <= horizon_frames <= 16:
            raise ValueError("horizon_frames must be between 1 and 16")
        if not np.isfinite(extra_margin_px) or extra_margin_px < 0.0:
            raise ValueError("extra_margin_px must be finite and non-negative")
        self.horizon_frames = int(horizon_frames)
        self.extra_margin_px = float(extra_margin_px)
        self.least_risk_fallback = bool(least_risk_fallback)
        self.fallback_count = 0

    def apply(self, raw_frame, base_mask: np.ndarray) -> tuple[np.ndarray, bool]:
        base_mask = np.asarray(base_mask, dtype=np.bool_)
        if base_mask.shape != (19,):
            raise ValueError("expected a 19-action base mask")
        native_mask = np.asarray(
            raw_frame.regular_bullet_action_mask(
                self.horizon_frames, self.extra_margin_px
            ),
            dtype=np.bool_,
        )
        if native_mask.shape != base_mask.shape:
            raise ValueError("native regular-bullet mask has the wrong shape")
        combined = base_mask & native_mask
        # When boundaries/resources remove every horizon-safe choice, keep the
        # policy inside the subset with the latest projected collision.  A
        # fail-open fallback would knowingly restore more immediate collisions.
        # A legal bomb is included by the native mask and therefore prevents
        # this movement-only fallback when present.
        if not combined.any():
            if not self.least_risk_fallback:
                return base_mask.copy(), False
            survival = np.asarray(
                raw_frame.regular_bullet_action_survival_frames(
                    self.horizon_frames, self.extra_margin_px
                ),
                dtype=np.int16,
            )
            if survival.shape != base_mask.shape:
                raise ValueError(
                    "native regular-bullet survival vector has the wrong shape"
                )
            legal_survival = np.where(base_mask, survival, -1)
            longest = int(legal_survival.max())
            if longest < 0:
                return base_mask.copy(), False
            combined = base_mask & (survival == longest)
            self.fallback_count += 1
            return combined, bool(np.any(base_mask & ~combined))
        intervened = bool(np.any(base_mask & ~combined))
        return combined, intervened


class DeathbombShield:
    """Reserve bombs, then force one inside TH05's audited eight-frame window."""

    def apply(self, raw_frame, base_mask: np.ndarray) -> tuple[np.ndarray, bool]:
        base_mask = np.asarray(base_mask, dtype=np.bool_)
        if base_mask.shape != (19,):
            raise ValueError("expected a 19-action base mask")
        if raw_frame.deathbomb_window_active():
            if not base_mask[18]:
                return base_mask.copy(), False
            mask = np.zeros_like(base_mask)
            mask[18] = True
            return mask, True

        # The current categorical policy often spends all bombs before a hit.
        # Reserving them makes the later exact cancellation executable. Keep a
        # bomb feasible when an upstream shield has already ruled out every
        # movement action; composition must never create an empty action set.
        if not base_mask[18]:
            return base_mask.copy(), False
        if not base_mask[:18].any():
            return base_mask.copy(), False
        mask = base_mask.copy()
        mask[18] = False
        return mask, False


__all__ = ["AuditedRegularBulletShield", "DeathbombShield", "EmergencyBombShield"]
