import unittest

import numpy as np

from pc98rl.env import ACTION_DIM, TH05_KINEMATICS
from pc98rl.model import ENTITY_COUNT, ENTITY_DIM, FEATURE_DIM, GLOBAL_DIM
from pc98rl.safety import EmergencyBombShield


class EmergencyBombShieldTest(unittest.TestCase):
    def setUp(self):
        self.shield = EmergencyBombShield(
            TH05_KINEMATICS, clearance_px=10.0, horizon_steps=6.0
        )
        self.observation = np.zeros(FEATURE_DIM, dtype=np.float32)
        for start in (GLOBAL_DIM, GLOBAL_DIM + ENTITY_COUNT * ENTITY_DIM):
            self.observation[
                start + ENTITY_DIM - 1 : start + ENTITY_COUNT * ENTITY_DIM : ENTITY_DIM
            ] = 1.0
        self.observation[10] = 3.0 / 8.0
        self.mask = np.ones(ACTION_DIM, dtype=np.bool_)

    def _set_collision_course(self) -> None:
        bullet = GLOBAL_DIM + ENTITY_COUNT * ENTITY_DIM
        self.observation[bullet] = 24.0 / TH05_KINEMATICS.position_scale[0]
        self.observation[bullet + 2] = -1.0
        self.observation[bullet + 5] = 1.0
        self.observation[bullet + 6] = 24.0 / TH05_KINEMATICS.distance_scale

    def test_forces_only_bomb_for_imminent_collision(self):
        self._set_collision_course()
        mask, intervened = self.shield.apply(self.observation, self.mask)
        self.assertTrue(intervened)
        self.assertEqual(np.flatnonzero(mask).tolist(), [18])

    def test_does_not_bomb_without_resources_or_while_invincible(self):
        self._set_collision_course()
        self.observation[10] = 0.0
        _, intervened = self.shield.apply(self.observation, self.mask)
        self.assertFalse(intervened)
        self.observation[10] = 3.0 / 8.0
        self.observation[11] = 1.0
        _, intervened = self.shield.apply(self.observation, self.mask)
        self.assertFalse(intervened)

    def test_receding_bullet_does_not_trigger(self):
        self._set_collision_course()
        bullet = GLOBAL_DIM + ENTITY_COUNT * ENTITY_DIM
        self.observation[bullet + 2] = 1.0
        _, intervened = self.shield.apply(self.observation, self.mask)
        self.assertFalse(intervened)

    def test_batched_masks_preserve_safe_rows(self):
        dangerous = self.observation.copy()
        self._set_collision_course()
        dangerous = self.observation.copy()
        safe = dangerous.copy()
        safe[11] = 1.0
        masks, interventions = self.shield.apply(
            np.stack((dangerous, safe)), np.stack((self.mask, self.mask))
        )
        np.testing.assert_array_equal(interventions, [True, False])
        self.assertEqual(np.flatnonzero(masks[0]).tolist(), [18])
        self.assertTrue(masks[1].all())


if __name__ == "__main__":
    unittest.main()
