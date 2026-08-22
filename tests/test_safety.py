import unittest

import numpy as np
import torch

from pc98rl.distributions import MaskedCategorical
from pc98rl.env import TH05_ACTIONS, TH05_CONSTRAINTS
from pc98rl.model import FEATURE_DIM


class AdapterConstraintTest(unittest.TestCase):
    def observation(self) -> np.ndarray:
        observation = np.zeros(FEATURE_DIM, dtype=np.float32)
        observation[4:8] = 0.5
        observation[10] = 0.25
        return observation

    def test_resource_and_boundary_constraints(self):
        observation = self.observation()
        observation[4] = 0.0
        observation[10] = 0.0
        mask = TH05_CONSTRAINTS.valid_actions(observation)
        self.assertFalse(mask[-1])
        for action, descriptor in enumerate(TH05_ACTIONS[:-1]):
            self.assertEqual(bool(mask[action]), descriptor.move_x >= 0)
        self.assertTrue(mask[0])
        self.assertTrue(mask[9])

    def test_batched_mask_shape(self):
        observations = np.stack((self.observation(), self.observation()))
        mask = TH05_CONSTRAINTS.valid_actions(observations)
        self.assertEqual(mask.shape, (2, 19))
        self.assertTrue(mask.all())


class MaskedCategoricalTest(unittest.TestCase):
    def test_invalid_action_has_zero_probability(self):
        logits = torch.zeros(2, 4)
        valid = torch.tensor([[True, False, True, False], [False, True, True, True]])
        distribution = MaskedCategorical(logits=logits, valid_mask=valid)
        self.assertTrue(torch.equal(distribution.probs[~valid], torch.zeros(3)))
        self.assertTrue(torch.allclose(distribution.removed_probability_mass, torch.tensor([0.5, 0.25])))
        samples = torch.stack([distribution.sample() for _ in range(100)])
        self.assertTrue(valid.gather(1, samples.T).all())

    def test_rejects_empty_mask(self):
        with self.assertRaises(ValueError):
            MaskedCategorical(
                logits=torch.zeros(1, 3), valid_mask=torch.zeros(1, 3, dtype=torch.bool)
            )


if __name__ == "__main__":
    unittest.main()

