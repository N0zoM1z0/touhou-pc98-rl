import unittest

import torch

from pc98rl.distill import (
    conservative_distillation_loss,
    conservative_outcome_loss,
    elite_episode_weight,
    standardized_outcome_advantages,
    trajectory_outcome_score,
)


class ConservativeDistillationTest(unittest.TestCase):
    def test_resource_weight_prefers_fewer_bombs(self):
        self.assertEqual(elite_episode_weight(0, 0.7), 1.0)
        self.assertGreater(elite_episode_weight(1, 0.7), elite_episode_weight(2, 0.7))

    def test_identical_policy_has_zero_anchor_kl(self):
        logits = torch.tensor([[1.0, 0.0, -1.0], [0.2, 0.3, 0.4]], requires_grad=True)
        actions = torch.tensor([0, 2])
        masks = torch.ones_like(logits, dtype=torch.bool)
        valid = torch.tensor([True, True])
        loss, metrics = conservative_distillation_loss(
            logits,
            logits.detach().clone(),
            actions,
            masks,
            valid,
            anchor_coefficient=0.5,
        )
        self.assertAlmostEqual(float(metrics["anchor_kl"]), 0.0, places=6)
        self.assertGreater(float(loss.detach()), 0.0)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_invalid_and_overridden_actions_do_not_contribute(self):
        student = torch.zeros(2, 3, requires_grad=True)
        teacher = torch.zeros_like(student)
        actions = torch.tensor([0, 1])
        masks = torch.tensor([[True, False, True], [True, True, True]])
        valid = torch.tensor([True, False])
        loss, _ = conservative_distillation_loss(
            student,
            teacher,
            actions,
            masks,
            valid,
            anchor_coefficient=0.0,
        )
        self.assertAlmostEqual(
            float(loss.detach()), float(torch.log(torch.tensor(2.0))), places=6
        )

    def test_outcome_score_requires_completion_and_prefers_resources(self):
        perfect = trajectory_outcome_score(
            no_miss_success=True, deaths=0, bombs_used=0
        )
        bombed = trajectory_outcome_score(
            no_miss_success=True, deaths=0, bombs_used=2
        )
        timeout = trajectory_outcome_score(
            no_miss_success=False, deaths=0, bombs_used=0
        )
        miss = trajectory_outcome_score(
            no_miss_success=False, deaths=1, bombs_used=0
        )
        self.assertGreater(perfect, bombed)
        self.assertGreater(bombed, timeout)
        self.assertGreater(timeout, miss)

    def test_standardized_advantages_are_centered(self):
        advantages = standardized_outcome_advantages(torch.tensor([1.0, 1.0, 0.0]))
        self.assertAlmostEqual(float(advantages.mean()), 0.0, places=6)
        self.assertAlmostEqual(float(advantages.square().mean()), 1.0, places=6)

    def test_outcome_loss_moves_positive_and_negative_actions_apart(self):
        student = torch.zeros(2, 2, requires_grad=True)
        behavior = torch.zeros_like(student)
        actions = torch.tensor([0, 1])
        masks = torch.ones_like(student, dtype=torch.bool)
        valid = torch.ones(2, dtype=torch.bool)
        advantages = torch.tensor([1.0, -1.0])
        loss, metrics = conservative_outcome_loss(
            student,
            behavior,
            actions,
            masks,
            valid,
            advantages,
            clip_ratio=0.2,
            anchor_coefficient=0.1,
        )
        loss.backward()
        self.assertLess(float(student.grad[0, 0]), 0.0)
        self.assertGreater(float(student.grad[1, 1]), 0.0)
        self.assertAlmostEqual(float(metrics["anchor_kl"]), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
