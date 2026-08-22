"""Cheap model-based safety teacher for compact TH05 observations."""

from __future__ import annotations

import numpy as np

from .model import ENTITY_COUNT, ENTITY_DIM, GLOBAL_DIM


_MOVES = np.asarray(
    [
        (0, 0),
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    ],
    dtype=np.float32,
)
_MOVES[5:] /= np.sqrt(2.0)


class SafetyHeuristic:
    """Experimental short-horizon safety baseline.

    Live evaluation found this policy worse than the random and PPO baselines,
    mainly because independent nearest-bullet predictions miss interactions
    between patterns.  It is retained as a negative ablation and is not used to
    warm up the PPO policy.
    """

    def __init__(self) -> None:
        self._bomb_cooldown = 0

    @staticmethod
    def _entities(features: np.ndarray) -> np.ndarray:
        projectile_start = GLOBAL_DIM
        bullet_start = projectile_start + ENTITY_COUNT * ENTITY_DIM
        projectile = features[
            projectile_start:bullet_start
        ].reshape(ENTITY_COUNT, ENTITY_DIM)
        bullet = features[
            bullet_start : bullet_start + ENTITY_COUNT * ENTITY_DIM
        ].reshape(ENTITY_COUNT, ENTITY_DIM)
        entities = np.concatenate((projectile, bullet), axis=0)
        present = (np.abs(entities[:, :6]).sum(axis=1) > 1e-7) | (
            entities[:, 6] < 1.0 - 1e-7
        )
        return entities[present]

    def act(self, features: np.ndarray) -> int:
        features = np.asarray(features, dtype=np.float32)
        entities = self._entities(features)
        self._bomb_cooldown = max(0, self._bomb_cooldown - 1)

        player_x = 8.0 + features[0] * 368.0
        player_y = 8.0 + features[1] * 344.0
        if len(entities):
            relative = np.stack(
                (entities[:, 0] * 384.0, entities[:, 1] * 368.0), axis=1
            )
            velocity = entities[:, 2:4] * 12.0
        else:
            relative = np.empty((0, 2), dtype=np.float32)
            velocity = np.empty((0, 2), dtype=np.float32)

        horizons = np.asarray((0.0, 1.5, 3.0, 5.0, 8.0), dtype=np.float32)
        imminent = np.inf
        imminent_time = np.inf
        if len(entities):
            projected = relative[:, None, :] + velocity[:, None, :] * horizons[None, :, None]
            sampled_distance = np.linalg.norm(projected, axis=2)
            sampled_index = np.unravel_index(np.argmin(sampled_distance), sampled_distance.shape)
            imminent = float(sampled_distance[sampled_index])
            imminent_time = float(horizons[sampled_index[1]])
            speed_sq = np.square(velocity).sum(axis=1).clip(min=1e-6)
            closest_time = np.clip(
                -np.sum(relative * velocity, axis=1) / speed_sq, 0.0, 24.0
            )
            closest = relative + velocity * closest_time[:, None]
            nearest_index = int(np.argmin(np.linalg.norm(closest, axis=1)))
            continuous_imminent = float(np.linalg.norm(closest[nearest_index]))
            if continuous_imminent < imminent:
                imminent = continuous_imminent
                imminent_time = float(closest_time[nearest_index])

        # Bomb only when a collision is predicted very soon.  The cooldown
        # prevents consuming all bombs while the previous blast is active.
        if (
            features[10] > 0.01
            and features[11] < 0.5
            and imminent < 18.0
            and imminent_time < 8.0
            and not self._bomb_cooldown
        ):
            self._bomb_cooldown = 45
            return 18

        precise = imminent < 58.0
        speed = 2.0 if precise else 4.0
        scores = np.zeros(len(_MOVES), dtype=np.float32)
        for move_index, direction in enumerate(_MOVES):
            player_velocity = direction * speed
            if len(entities):
                relative_velocity = velocity - player_velocity
                speed_sq = np.square(relative_velocity).sum(axis=1).clip(min=1e-6)
                closest_time = np.clip(
                    -np.sum(relative * relative_velocity, axis=1) / speed_sq,
                    0.0,
                    24.0,
                )
                closest = relative + relative_velocity * closest_time[:, None]
                closest_distance = np.linalg.norm(closest, axis=1)
                scores[move_index] += 5.0 * np.sum(
                    np.exp(-np.square(closest_distance / 12.0))
                    * np.exp(-closest_time / 12.0)
                )
                projected = (
                    relative[:, None, :]
                    + (velocity[:, None, :] - player_velocity) * horizons[None, :, None]
                )
                distance = np.linalg.norm(projected, axis=2)
                time_weight = np.asarray((1.6, 1.3, 1.0, 0.7, 0.4), dtype=np.float32)
                scores[move_index] += np.sum(
                    np.exp(-np.square(distance / 15.0)) * time_weight[None, :]
                )

            predicted_x = player_x + player_velocity[0] * 8.0
            predicted_y = player_y + player_velocity[1] * 8.0
            wall_margin = min(
                predicted_x - 8.0,
                376.0 - predicted_x,
                predicted_y - 8.0,
                352.0 - predicted_y,
            )
            scores[move_index] += 8.0 * np.exp(-max(wall_margin, 0.0) / 12.0)

            # Reserve a lower-center operating band.  Pure short-horizon risk
            # minimization otherwise walks into a wall, where the next pattern
            # has no escape route.  Out-of-band moves remain available only if
            # every candidate is bad, but receive a dominating cost.
            if predicted_x < 24.0 or predicted_x > 360.0:
                scores[move_index] += 1_000.0
            if predicted_y < 215.0 or predicted_y > 326.0:
                scores[move_index] += 1_000.0

            # Small strategic terms break ties: remain low on the playfield and
            # align with an active boss without overpowering collision risk.
            target_y = 0.78
            y_error = ((predicted_y - 8.0) / 344.0 - target_y) / 0.16
            scores[move_index] += 0.8 * y_error * y_error
            x_error = ((predicted_x - 8.0) / 368.0 - 0.5) / 0.45
            scores[move_index] += 0.04 * x_error * x_error
            if features[17] > 0.5:
                boss_dx = features[19] * 500.0
                scores[move_index] += 0.03 * abs(boss_dx - player_velocity[0] * 8.0) / 100.0

        movement = int(np.argmin(scores))
        return movement + (9 if precise else 0)


__all__ = ["SafetyHeuristic"]
