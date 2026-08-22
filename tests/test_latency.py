import unittest

from pc98rl.latency import deadline_utilization, latency_summary


class LatencySummaryTest(unittest.TestCase):
    def test_reports_tail_and_maximum(self):
        summary = latency_summary([1.0, 2.0, 3.0, 20.0])
        self.assertEqual(summary["samples"], 4)
        self.assertEqual(summary["max_ms"], 20.0)
        self.assertGreater(summary["p99_ms"], summary["p95_ms"])

    def test_deadline_utilization(self):
        self.assertAlmostEqual(deadline_utilization(9.0, 36.0), 0.25)

    def test_rejects_empty_samples(self):
        with self.assertRaises(ValueError):
            latency_summary([])


if __name__ == "__main__":
    unittest.main()
