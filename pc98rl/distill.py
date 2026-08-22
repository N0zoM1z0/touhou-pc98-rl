"""Offline conservative distillation helpers for the real-time actor."""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F


def elite_episode_weight(bombs_used: int, decay: float) -> float:
    """Prefer successful trajectories that required fewer emergency resources."""
    if bombs_used < 0:
        raise ValueError("bombs_used must be non-negative")
    if not math.isfinite(decay) or decay < 0.0:
        raise ValueError("decay must be finite and non-negative")
    return math.exp(-decay * bombs_used)


def conservative_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    actions: torch.Tensor,
    action_masks: torch.Tensor,
    valid: torch.Tensor,
    *,
    anchor_coefficient: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Reinforce elite actions while anchoring the full teacher distribution.

    Mid-transaction safety overrides are excluded with ``valid``. Invalid game
    actions receive zero probability in both policies, exactly as in PPO.
    """
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student and teacher logits must have identical shapes")
    if student_logits.shape != action_masks.shape:
        raise ValueError("action masks must match logits")
    if actions.shape != student_logits.shape[:-1] or valid.shape != actions.shape:
        raise ValueError("actions and valid must match the logits batch shape")
    if anchor_coefficient < 0.0 or not math.isfinite(anchor_coefficient):
        raise ValueError("anchor_coefficient must be finite and non-negative")
    if not torch.any(valid):
        raise ValueError("distillation batch has no attributable policy actions")
    if not torch.all(action_masks.any(dim=-1)):
        raise ValueError("every state must retain at least one valid action")

    invalid_logit = torch.finfo(student_logits.dtype).min
    masked_student = student_logits.masked_fill(~action_masks, invalid_logit)
    masked_teacher = teacher_logits.masked_fill(~action_masks, invalid_logit)
    student_log_probabilities = F.log_softmax(masked_student, dim=-1)
    with torch.no_grad():
        teacher_log_probabilities = F.log_softmax(masked_teacher, dim=-1)
        teacher_probabilities = teacher_log_probabilities.exp()

    elite_nll = -student_log_probabilities.gather(
        -1, actions.unsqueeze(-1)
    ).squeeze(-1)
    anchor_kl = torch.where(
        action_masks,
        teacher_probabilities
        * (teacher_log_probabilities - student_log_probabilities),
        torch.zeros_like(teacher_probabilities),
    ).sum(dim=-1)
    elite_nll = elite_nll[valid].mean()
    anchor_kl = anchor_kl[valid].mean()
    loss = elite_nll + anchor_coefficient * anchor_kl
    entropy = -torch.where(
        action_masks,
        student_log_probabilities.exp() * student_log_probabilities,
        torch.zeros_like(student_log_probabilities),
    ).sum(dim=-1)[valid].mean()
    return loss, {
        "elite_nll": elite_nll.detach(),
        "anchor_kl": anchor_kl.detach(),
        "entropy": entropy.detach(),
    }


__all__ = ["conservative_distillation_loss", "elite_episode_weight"]
