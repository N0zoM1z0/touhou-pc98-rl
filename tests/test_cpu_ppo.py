import unittest

import numpy as np

from pc98rl.ppo import _as_sequences, _vector_gae


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


if __name__ == "__main__":
    unittest.main()
