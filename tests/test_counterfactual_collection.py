import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.collect_counterfactual_branches import (
    DATASET_FORMAT,
    _anchor_arrays,
    _write_dataset,
)


def _outcome(action, collision=None, kind=None):
    return {
        "action": action,
        "collision_native_frames": collision,
        "collision_kind": kind,
    }


class CounterfactualCollectionTest(unittest.TestCase):
    def test_anchor_arrays_distinguish_illegal_safe_and_collision(self):
        anchor = {
            "outcomes": [
                _outcome(0, 2, "pending_hit"),
                _outcome(1),
                _outcome(2, None, "structurally_illegal"),
            ]
        }
        mask, frames = _anchor_arrays(anchor)
        self.assertEqual(mask[:3].tolist(), [True, True, False])
        self.assertEqual(frames[:3].tolist(), [2, -1, -2])

    def test_dataset_keeps_one_trajectory_and_actor_features(self):
        outcomes = [
            _outcome(action, 2 if action == 0 else None)
            for action in range(18)
        ]
        anchor = {
            "_actor_feature": np.arange(128, dtype=np.float32),
            "behavior_logits": [0.0] * 19,
            "stage_frame": 439,
            "search_decision": 188,
            "behavior_action": 3,
            "unsafe_actions_at_trigger_horizon": 7,
            "projected_survival_frames": [0] * 18,
            "outcomes": outcomes,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "seed_11.npz"
            _write_dataset(
                path,
                anchors=[anchor],
                seed=11,
                checkpoint_sha256="abc",
                trigger_horizon=2,
                continuation_horizon=6,
                continuation_decisions=16,
                native_frames=2,
                trajectory_policy="sample",
            )
            with np.load(path) as dataset:
                self.assertEqual(str(dataset["format"]), DATASET_FORMAT)
                self.assertEqual(str(dataset["trajectory_id"]), "seed_11")
                self.assertEqual(dataset["actor_features"].shape, (1, 128))
                self.assertEqual(dataset["behavior_logits"].shape, (1, 19))
                self.assertEqual(dataset["collision_risk"][0, :2].tolist(), [1.0, 0.0])
                self.assertFalse(bool(dataset["action_masks"][0, 18]))
                self.assertEqual(str(dataset["trajectory_policy"]), "sample")


if __name__ == "__main__":
    unittest.main()
