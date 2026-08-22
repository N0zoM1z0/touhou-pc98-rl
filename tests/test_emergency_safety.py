import unittest

import numpy as np

from pc98rl.env import ACTION_DIM, TH05_KINEMATICS
from pc98rl.model import ENTITY_COUNT, ENTITY_DIM, FEATURE_DIM, GLOBAL_DIM
from pc98rl.safety import AuditedRegularBulletShield, EmergencyBombShield


class _FakeRawFrame:
    def __init__(self, mask):
        self.mask = mask
        self.calls = []

    def regular_bullet_action_mask(self, horizon_frames, extra_margin_px):
        self.calls.append((horizon_frames, extra_margin_px))
        return self.mask


class AuditedRegularBulletShieldTest(unittest.TestCase):
    def test_combines_native_and_structural_masks(self):
        base = np.ones(19, dtype=np.bool_)
        base[1] = False
        native = np.ones(19, dtype=np.bool_)
        native[2] = False
        frame = _FakeRawFrame(native)
        shield = AuditedRegularBulletShield(horizon_frames=3, extra_margin_px=0.5)
        mask, intervened = shield.apply(frame, base)
        self.assertTrue(intervened)
        self.assertFalse(mask[1])
        self.assertFalse(mask[2])
        self.assertEqual(frame.calls, [(3, 0.5)])

    def test_fails_open_if_combination_is_empty(self):
        base = np.zeros(19, dtype=np.bool_)
        base[4] = True
        native = np.ones(19, dtype=np.bool_)
        native[4] = False
        mask, intervened = AuditedRegularBulletShield().apply(
            _FakeRawFrame(native), base
        )
        np.testing.assert_array_equal(mask, base)
        self.assertFalse(intervened)


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
