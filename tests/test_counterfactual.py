import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from pc98rl.counterfactual import (
    DATASET_FORMAT,
    counterfactual_policy_metrics,
    load_counterfactual_group,
    validate_disjoint_groups,
)


def _dataset(path: Path, trajectory_id: str, risky_action: int = 0):
    risk = np.zeros((1, 19), dtype=np.float32)
    risk[0, risky_action] = 1.0
    mask = np.zeros((1, 19), dtype=np.bool_)
    mask[0, :3] = True
    np.savez_compressed(
        path,
        format=np.asarray(DATASET_FORMAT),
        trajectory_id=np.asarray(trajectory_id),
        checkpoint_sha256=np.asarray("source"),
        actor_features=np.zeros((1, 128), dtype=np.float32),
        behavior_logits=np.zeros((1, 19), dtype=np.float32),
        action_masks=mask,
        collision_risk=risk,
    )


class CounterfactualDatasetTest(unittest.TestCase):
    def test_load_and_policy_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "one.npz"
            _dataset(path, "one")
            group = load_counterfactual_group([path])
        metrics = counterfactual_policy_metrics(group.behavior_logits, group)
        self.assertEqual(group.anchors, 1)
        self.assertAlmostEqual(metrics["expected_collision_risk"], 1.0 / 3.0)
        self.assertEqual(metrics["argmax_collision_fraction"], 1.0)
        self.assertAlmostEqual(metrics["behavior_kl"], 0.0)

    def test_groups_must_be_disjoint_by_trajectory_not_anchor(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.npz"
            second = Path(directory) / "second.npz"
            _dataset(first, "shared")
            _dataset(second, "shared", risky_action=1)
            train = load_counterfactual_group([first])
            selection = load_counterfactual_group([second])
            with self.assertRaisesRegex(ValueError, "both train and selection"):
                validate_disjoint_groups(train=train, selection=selection)

    def test_lower_risky_logit_reduces_causal_risk_mass(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "one.npz"
            _dataset(path, "one")
            group = load_counterfactual_group([path])
        student = torch.zeros_like(group.behavior_logits)
        student[0, 0] = -3.0
        metrics = counterfactual_policy_metrics(student, group)
        self.assertLess(metrics["expected_collision_risk"], 0.03)
        self.assertEqual(metrics["argmax_collision_fraction"], 0.0)


if __name__ == "__main__":
    unittest.main()
