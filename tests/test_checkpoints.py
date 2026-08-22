import tempfile
import unittest
from pathlib import Path

import torch

from pc98rl.checkpoints import deployment_checkpoint


class DeploymentCheckpointTest(unittest.TestCase):
    def test_strips_training_and_teacher_state(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.pt"
            source.write_bytes(b"source")
            saved = {
                "model": {"weight": torch.ones(2)},
                "args": {"regular_bullet_safety_horizon": 6},
                "optimizer": {"state": "large"},
                "future_miss_head": {"weight": torch.ones(1)},
                "future_safety_distillation": {
                    "risk_head": {"weight": torch.ones(1)},
                    "trajectories": ["private/a.npz", "private/b.npz"],
                    "epoch": 4,
                },
            }
            result = deployment_checkpoint(
                saved,
                source=source,
                argument_overrides={"regular_bullet_least_risk_fallback": True},
            )

        self.assertNotIn("optimizer", result)
        self.assertNotIn("future_miss_head", result)
        self.assertNotIn("risk_head", result["future_safety_distillation"])
        self.assertNotIn("trajectories", result["future_safety_distillation"])
        self.assertEqual(
            result["future_safety_distillation"]["trajectory_count"], 2
        )
        self.assertTrue(result["args"]["regular_bullet_least_risk_fallback"])
        self.assertTrue(result["deployment"]["offline_teacher_stripped"])


if __name__ == "__main__":
    unittest.main()
