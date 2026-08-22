"""Game-independent contracts consumed by the reinforcement-learning core."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot


@dataclass(frozen=True)
class KinematicSpec:
    """Units needed to derive relative-motion features from an adapter schema.

    Adapters normalize positions and velocities for storage.  Supplying their
    physical scales here lets the shared model perform the same geometry for
    any game without embedding a playfield size or speed convention in the
    learner.
    """

    position_scale: tuple[float, float]
    velocity_scale: tuple[float, float]
    horizon_steps: float

    def __post_init__(self) -> None:
        values = (*self.position_scale, *self.velocity_scale, self.horizon_steps)
        if any(value <= 0.0 for value in values):
            raise ValueError("kinematic scales and horizon must be positive")

    @property
    def distance_scale(self) -> float:
        return hypot(*self.position_scale)


@dataclass(frozen=True)
class ActionDescriptor:
    """Semantic action metadata suitable for cross-game policy heads."""

    move_x: int = 0
    move_y: int = 0
    primary: bool = False
    secondary: bool = False
    bomb: bool = False

    def __post_init__(self) -> None:
        if self.move_x not in (-1, 0, 1) or self.move_y not in (-1, 0, 1):
            raise ValueError("movement components must be -1, 0, or 1")


class ConstraintProvider:
    """Adapter-side interface for masks that are exact under game rules."""

    def valid_actions(self, observation):  # pragma: no cover - protocol method
        raise NotImplementedError

