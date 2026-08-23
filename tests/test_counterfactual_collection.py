import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from scripts.collect_counterfactual_branches import (
    DATASET_FORMAT,
    _anchor_arrays,
    _write_dataset,
    collect_trajectory_with_retries,
    movement_mask,
)


def _outcome(action, collision=None, kind=None):
    return {
        "action": action,
        "collision_native_frames": collision,
        "collision_kind": kind,
    }


class CounterfactualCollectionTest(unittest.TestCase):
    def test_movement_mask_always_excludes_bomb_for_nmnb(self):
        class Shield:
            def apply(self, raw_frame, mask):
                return mask, False

        mask, _ = movement_mask(Shield(), object(), np.ones(19, dtype=np.bool_))
        self.assertFalse(bool(mask[18]))

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

    def test_collection_retries_only_transient_timeout(self):
        attempts = 0

        def transient(task):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise TimeoutError("emulator stalled")
            Path(task["output"]).write_text("{}\n", encoding="utf-8")
            return {"trajectory_id": "seed_7"}

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "seed_7.json"
            with mock.patch(
                "scripts.collect_counterfactual_branches.collect_trajectory",
                side_effect=transient,
            ):
                result = collect_trajectory_with_retries(
                    {"seed": 7, "output": str(output), "trajectory_retries": 2}
                )
            self.assertEqual(attempts, 2)
            self.assertEqual(result["collection_attempts"], 2)
            self.assertEqual(
                json.loads(output.read_text())["collection_attempts"], 2
            )


if __name__ == "__main__":
    unittest.main()
