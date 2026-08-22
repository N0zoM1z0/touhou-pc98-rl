import unittest

import torch

from pc98rl.distill import conservative_distillation_loss, elite_episode_weight


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
        self.assertAlmostEqual(float(loss), float(torch.log(torch.tensor(2.0))), places=6)


if __name__ == "__main__":
    unittest.main()
