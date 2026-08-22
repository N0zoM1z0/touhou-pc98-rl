import unittest

import torch

from pc98rl.offline_risk import (
    balanced_event_loss,
    future_event_targets,
    risk_adjusted_distillation_loss,
)


class OfflineRiskTest(unittest.TestCase):
    def test_future_event_excludes_current_preemption(self):
        events = torch.tensor([False, True, False, False])
        targets, valid = future_event_targets(events, (1, 2), terminal=False)
        self.assertEqual(targets.tolist(), [[1, 1], [0, 0], [0, 0], [0, 0]])
        self.assertEqual(
            valid.tolist(), [[True, True], [True, True], [True, False], [False, False]]
        )

    def test_terminal_suffix_is_observed_safe(self):
        _, valid = future_event_targets(
            torch.zeros(3, dtype=torch.bool), (2,), terminal=True
        )
        self.assertTrue(valid.all())

    def test_balanced_loss_is_finite_without_positive_labels(self):
        logits = torch.zeros(4, 2, requires_grad=True)
        targets = torch.zeros_like(logits)
        valid = torch.ones_like(logits, dtype=torch.bool)
        loss = balanced_event_loss(logits, targets, valid)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_risk_teacher_suppresses_risky_action(self):
        student = torch.zeros(1, 3, requires_grad=True)
        behavior = torch.zeros_like(student)
        masks = torch.ones_like(student, dtype=torch.bool)
        risk = torch.tensor([[0.0, 1.0, 0.0]])
        loss, _ = risk_adjusted_distillation_loss(
            student,
            behavior,
            masks,
            risk,
            risk_scale=2.0,
            anchor_coefficient=0.1,
        )
        loss.backward()
        self.assertGreater(float(student.grad[0, 1]), 0.0)


if __name__ == "__main__":
    unittest.main()
