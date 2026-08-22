import unittest

import torch

from pc98rl.contracts import KinematicSpec
from pc98rl.model import (
    ENTITY_COUNT,
    ENTITY_DIM,
    FEATURE_DIM,
    EntityActorCritic,
    EntitySetEncoder,
    FutureMissHead,
    add_kinematic_features,
)


TEST_KINEMATICS = KinematicSpec(
    position_scale=(384.0, 368.0),
    velocity_scale=(12.0, 12.0),
    horizon_steps=60.0,
)


class EntityActorCriticTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.model = EntityActorCritic()

    def test_step_shapes(self):
        features = torch.zeros(3, FEATURE_DIM)
        logits, values, hidden = self.model.forward_step(features)
        self.assertEqual(logits.shape, (3, 19))
        self.assertEqual(values.shape, (3, 3))
        self.assertEqual(hidden.shape, (1, 3, 128))
        self.assertTrue(torch.isfinite(logits).all())

    def test_sequence_shapes_and_terminal_reset(self):
        features = torch.randn(2, 5, FEATURE_DIM)
        hidden = torch.zeros(1, 2, 128)
        dones = torch.zeros(2, 5)
        dones[0, 2] = 1
        logits, values = self.model.forward_sequence(features, hidden, dones)
        self.assertEqual(logits.shape, (10, 19))
        self.assertEqual(values.shape, (10, 3))

    def test_training_only_future_miss_head_shapes(self):
        recurrent = torch.randn(7, self.model.hidden_size)
        actions = torch.arange(7) % 19
        head = FutureMissHead(self.model.hidden_size, action_dim=19, horizon_count=2)
        self.assertEqual(head(recurrent, actions).shape, (7, 2))

    def test_all_padding_is_finite(self):
        features = torch.zeros(4, FEATURE_DIM)
        for start in (37, 37 + ENTITY_COUNT * ENTITY_DIM):
            features[:, start + ENTITY_DIM - 1 : start + ENTITY_COUNT * ENTITY_DIM : ENTITY_DIM] = 1
        logits, values, _ = self.model.forward_step(features)
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.isfinite(values).all())

    def test_distance_sentinel_is_not_treated_as_an_entity(self):
        encoder = EntitySetEncoder()
        tokens = torch.zeros(2, ENTITY_COUNT, ENTITY_DIM)
        tokens[..., -1] = 1.0
        with torch.no_grad():
            output = encoder(tokens)
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.equal(output[0], output[1]))

    def test_analytic_geometry_recovers_collision_course(self):
        tokens = torch.zeros(1, 1, ENTITY_DIM)
        tokens[..., 0] = 120.0 / TEST_KINEMATICS.position_scale[0]
        tokens[..., 2] = -1.0
        tokens[..., 6] = 120.0 / (384.0**2 + 368.0**2) ** 0.5
        augmented = add_kinematic_features(
            tokens, torch.zeros(1, 2), TEST_KINEMATICS
        )
        self.assertEqual(augmented.shape, (1, 1, ENTITY_DIM + 5))
        self.assertAlmostEqual(float(augmented[0, 0, 7]), -0.5, places=6)
        self.assertAlmostEqual(float(augmented[0, 0, 9]), 0.5, places=6)
        self.assertAlmostEqual(float(augmented[0, 0, 10]), 1.0 / 6.0, places=6)
        self.assertAlmostEqual(float(augmented[0, 0, 11]), 0.0, places=6)

    def test_analytic_geometry_model_handles_padding(self):
        model = EntityActorCritic(
            analytic_geometry=True, kinematic_spec=TEST_KINEMATICS
        )
        features = torch.zeros(2, FEATURE_DIM)
        for start in (37, 37 + ENTITY_COUNT * ENTITY_DIM):
            features[:, start + ENTITY_DIM - 1 : start + ENTITY_COUNT * ENTITY_DIM : ENTITY_DIM] = 1
        logits, values, _ = model.forward_step(features)
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.isfinite(values).all())

    def test_model_is_compact(self):
        parameter_count = sum(parameter.numel() for parameter in self.model.parameters())
        self.assertLess(parameter_count, 500_000)


if __name__ == "__main__":
    unittest.main()
