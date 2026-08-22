"""Offline conservative distillation helpers for the real-time actor."""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F


def trajectory_outcome_score(
    *,
    no_miss_success: bool,
    deaths: int,
    bombs_used: int,
    completion_reward: float = 1.0,
    death_penalty: float = 1.0,
    bomb_penalty: float = 0.05,
) -> float:
    """Map a complete deployment attempt to a conservative scalar utility.

    Completion is deliberately binary: censored survival is not confused with
    clearing the segment.  Deaths are always costly, while bomb use is only a
    small tie-breaker among otherwise successful trajectories.
    """
    if deaths < 0:
        raise ValueError("deaths must be non-negative")
    if bombs_used < 0:
        raise ValueError("bombs_used must be non-negative")
    coefficients = (completion_reward, death_penalty, bomb_penalty)
    if not all(math.isfinite(value) and value >= 0.0 for value in coefficients):
        raise ValueError("outcome coefficients must be finite and non-negative")
    return (
        completion_reward * float(no_miss_success)
        - death_penalty * deaths
        - bomb_penalty * bombs_used
    )


def standardized_outcome_advantages(scores: torch.Tensor) -> torch.Tensor:
    """Produce zero-mean episode advantages without unstable tiny batches."""
    if scores.ndim != 1 or len(scores) < 2:
        raise ValueError("at least two scalar episode scores are required")
    if not torch.isfinite(scores).all():
        raise ValueError("episode scores must be finite")
    centered = scores - scores.mean()
    scale = centered.square().mean().sqrt()
    if float(scale) < 1e-8:
        raise ValueError("episode outcomes have no variation")
    return centered / scale


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


def conservative_outcome_loss(
    student_logits: torch.Tensor,
    behavior_logits: torch.Tensor,
    actions: torch.Tensor,
    action_masks: torch.Tensor,
    valid: torch.Tensor,
    advantages: torch.Tensor,
    *,
    clip_ratio: float,
    anchor_coefficient: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Clipped offline policy improvement anchored to the behavior policy.

    Trajectories must have been sampled from ``behavior_logits``.  Positive
    outcome advantages reinforce attributable actions and negative advantages
    suppress them.  The teacher-to-student KL bounds changes to actions that
    were not sampled, keeping this small-data update conservative.
    """
    if student_logits.shape != behavior_logits.shape:
        raise ValueError("student and behavior logits must have identical shapes")
    if student_logits.shape != action_masks.shape:
        raise ValueError("action masks must match logits")
    expected_shape = student_logits.shape[:-1]
    if any(tensor.shape != expected_shape for tensor in (actions, valid, advantages)):
        raise ValueError("actions, valid, and advantages must match the logits batch shape")
    if not 0.0 < clip_ratio < 1.0 or not math.isfinite(clip_ratio):
        raise ValueError("clip_ratio must be finite and in (0, 1)")
    if anchor_coefficient < 0.0 or not math.isfinite(anchor_coefficient):
        raise ValueError("anchor_coefficient must be finite and non-negative")
    if not torch.any(valid):
        raise ValueError("outcome batch has no attributable policy actions")
    if not torch.all(action_masks.any(dim=-1)):
        raise ValueError("every state must retain at least one valid action")
    if not torch.isfinite(advantages).all():
        raise ValueError("advantages must be finite")

    invalid_logit = torch.finfo(student_logits.dtype).min
    masked_student = student_logits.masked_fill(~action_masks, invalid_logit)
    masked_behavior = behavior_logits.masked_fill(~action_masks, invalid_logit)
    student_log_probabilities = F.log_softmax(masked_student, dim=-1)
    with torch.no_grad():
        behavior_log_probabilities = F.log_softmax(masked_behavior, dim=-1)
        behavior_probabilities = behavior_log_probabilities.exp()

    student_action_log_prob = student_log_probabilities.gather(
        -1, actions.unsqueeze(-1)
    ).squeeze(-1)
    behavior_action_log_prob = behavior_log_probabilities.gather(
        -1, actions.unsqueeze(-1)
    ).squeeze(-1)
    ratio = torch.exp(student_action_log_prob - behavior_action_log_prob)
    unclipped = ratio * advantages
    clipped = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
    policy_loss = -torch.minimum(unclipped, clipped)[valid].mean()

    anchor_kl = torch.where(
        action_masks,
        behavior_probabilities
        * (behavior_log_probabilities - student_log_probabilities),
        torch.zeros_like(behavior_probabilities),
    ).sum(dim=-1)[valid].mean()
    entropy = -torch.where(
        action_masks,
        student_log_probabilities.exp() * student_log_probabilities,
        torch.zeros_like(student_log_probabilities),
    ).sum(dim=-1)[valid].mean()
    loss = policy_loss + anchor_coefficient * anchor_kl
    return loss, {
        "policy_loss": policy_loss.detach(),
        "anchor_kl": anchor_kl.detach(),
        "entropy": entropy.detach(),
        "mean_ratio": ratio[valid].mean().detach(),
        "clip_fraction": ((ratio - 1.0).abs() > clip_ratio)[valid].float().mean().detach(),
    }


__all__ = [
    "conservative_distillation_loss",
    "conservative_outcome_loss",
    "elite_episode_weight",
    "standardized_outcome_advantages",
    "trajectory_outcome_score",
]
