import unittest

import numpy as np
import torch

from pc98rl.ppo import _apply_checkpoint, _as_sequences, _vector_gae


class CpuPPOHelpersTest(unittest.TestCase):
    def test_vector_gae_does_not_bootstrap_across_terminal(self):
        rewards = np.zeros((2, 1, 3), dtype=np.float32)
        values = np.ones_like(rewards)
        dones = np.asarray([[1.0], [0.0]], dtype=np.float32)
        last_values = np.full((1, 3), 2.0, dtype=np.float32)
        advantages, returns = _vector_gae(
            rewards, values, dones, last_values, gamma=1.0, gae_lambda=1.0
        )
        np.testing.assert_allclose(advantages[0], -1.0)
        np.testing.assert_allclose(returns[0], 0.0)
        np.testing.assert_allclose(advantages[1], 1.0)

    def test_sequence_layout_keeps_each_environment_contiguous(self):
        time_env = np.arange(8).reshape(4, 2)
        sequences = _as_sequences(time_env, 2)
        np.testing.assert_array_equal(
            sequences, np.asarray([[0, 2], [4, 6], [1, 3], [5, 7]])
        )

    def test_weights_only_initialization_resets_optimizer_and_counters(self):
        source = torch.nn.Linear(2, 1)
        source_optimizer = torch.optim.Adam(source.parameters(), lr=0.01)
        source(torch.ones(1, 2)).sum().backward()
        source_optimizer.step()
        checkpoint = {
            "model": source.state_dict(),
            "optimizer": source_optimizer.state_dict(),
            "update": 9,
            "environment_steps": 1234,
        }

        target = torch.nn.Linear(2, 1)
        target_optimizer = torch.optim.Adam(target.parameters(), lr=0.123)
        update, environment_steps = _apply_checkpoint(
            target, target_optimizer, checkpoint, resume=False
        )

        self.assertEqual((update, environment_steps), (0, 0))
        self.assertEqual(target_optimizer.param_groups[0]["lr"], 0.123)
        self.assertEqual(target_optimizer.state, {})
        for source_parameter, target_parameter in zip(
            source.parameters(), target.parameters(), strict=True
        ):
            torch.testing.assert_close(source_parameter, target_parameter)


if __name__ == "__main__":
    unittest.main()
