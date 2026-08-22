import unittest

from scripts.compare_policies import summarize


class EvaluationSummaryTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
