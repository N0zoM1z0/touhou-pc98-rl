"""Policy distributions shared by rollout, optimization, and evaluation."""

from __future__ import annotations

import torch
from torch.distributions import Categorical


class MaskedCategorical(Categorical):
    """Categorical distribution with exact renormalization over valid actions."""

    def __init__(
        self,
        *,
        logits: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> None:
        if valid_mask is None:
            valid_mask = torch.ones_like(logits, dtype=torch.bool)
        else:
            valid_mask = valid_mask.to(device=logits.device, dtype=torch.bool)
            if valid_mask.shape != logits.shape:
                raise ValueError("action mask and logits must have identical shapes")
        if not torch.all(valid_mask.any(dim=-1)):
            raise ValueError("every state must leave at least one valid action")

        raw_probabilities = torch.softmax(logits, dim=-1)
        self.valid_mask = valid_mask
        self.removed_probability_mass = (
            raw_probabilities * (~valid_mask).to(raw_probabilities.dtype)
        ).sum(dim=-1)
        masked_logits = logits.masked_fill(~valid_mask, -torch.inf)
        super().__init__(logits=masked_logits)

    @property
    def mode(self) -> torch.Tensor:
        return self.logits.argmax(dim=-1)

