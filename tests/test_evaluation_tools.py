import unittest

from scripts.compare_policies import summarize
from scripts.select_checkpoint import candidate_rank, runtime_config_mismatches


class EvaluationSummaryTest(unittest.TestCase):
    def test_checkpoint_rank_prefers_no_miss_before_return(self):
        perfect = {
            "successes": 3,
            "no_miss_successes": 2,
            "selection_score": -1.0,
        }
        higher_return = {
            "successes": 3,
            "no_miss_successes": 1,
            "selection_score": 10.0,
        }
        self.assertGreater(candidate_rank(perfect), candidate_rank(higher_return))

    def test_checkpoint_rank_prefers_more_no_miss_over_more_raw_clears(self):
        safer = {
            "successes": 7,
            "no_miss_successes": 7,
            "selection_score": 0.0,
        }
        less_safe = {
            "successes": 8,
            "no_miss_successes": 6,
            "selection_score": 10.0,
        }
        self.assertGreater(candidate_rank(safer), candidate_rank(less_safe))

    def test_summary_preserves_vector_objectives(self):
        summary = summarize(
            [
                {
                    "scalar_return": 1.0,
                    "death_events": 0,
                    "success": True,
                    "raw_reward": [10.0, 2.0, -4.0],
                },
                {
                    "scalar_return": -1.0,
                    "death_events": 2,
                    "success": False,
                    "raw_reward": [0.0, 4.0, -2.0],
                },
            ]
        )
        self.assertEqual(summary["mean_return"], 0.0)
        self.assertEqual(summary["deaths"], 2)
        self.assertEqual(summary["successes"], 1)
        self.assertEqual(summary["mean_raw_reward"], [5.0, 3.0, -3.0])

    def test_runtime_mismatch_requires_an_explicit_override(self):
        arguments = [
            {"deathbomb_safety": False, "regular_bullet_safety_horizon": 6},
            {"deathbomb_safety": True, "regular_bullet_safety_horizon": 6},
        ]
        mismatches = runtime_config_mismatches(arguments, {})
        self.assertEqual(mismatches, {"deathbomb_safety": [False, True]})
        self.assertEqual(
            runtime_config_mismatches(
                arguments, {"deathbomb_safety": True}
            ),
            {},
        )


if __name__ == "__main__":
    unittest.main()
