"""Checkpoint transformations at the offline/online deployment boundary."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deployment_checkpoint(
    saved: dict[str, Any],
    *,
    source: Path,
    argument_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return an inference-only checkpoint without learner or teacher state."""
    if "model" not in saved or "args" not in saved:
        raise ValueError("checkpoint must contain model weights and runtime arguments")
    arguments = copy.deepcopy(saved["args"])
    arguments.update(argument_overrides or {})

    provenance = copy.deepcopy(saved.get("future_safety_distillation"))
    if provenance is not None:
        trajectories = provenance.pop("trajectories", [])
        source_checkpoint = provenance.pop("source", None)
        provenance.pop("risk_head", None)
        provenance["trajectory_count"] = len(trajectories)
        if source_checkpoint is not None:
            provenance["source_checkpoint"] = Path(source_checkpoint).name

    counterfactual_provenance = copy.deepcopy(
        saved.get("counterfactual_distillation")
    )
    if counterfactual_provenance is not None:
        train_trajectories = counterfactual_provenance.pop(
            "train_trajectories", []
        )
        selection_trajectories = counterfactual_provenance.pop(
            "selection_trajectories", []
        )
        source_checkpoint = counterfactual_provenance.pop("source", None)
        counterfactual_provenance["train_trajectory_count"] = len(
            train_trajectories
        )
        counterfactual_provenance["selection_trajectory_count"] = len(
            selection_trajectories
        )
        if source_checkpoint is not None:
            counterfactual_provenance["source_checkpoint"] = Path(
                source_checkpoint
            ).name

    deployment = {
        "format": "pc98rl-online-actor-v1",
        "source_checkpoint": source.name,
        "source_sha256": sha256_file(source),
        "training_state_stripped": True,
        "offline_teacher_stripped": True,
    }
    result = {
        "model": saved["model"],
        "args": arguments,
        "update": saved.get("update"),
        "environment_steps": saved.get("environment_steps"),
        "scenario": copy.deepcopy(saved.get("scenario")),
        "initialization": copy.deepcopy(saved.get("initialization")),
        "deployment": deployment,
    }
    if provenance is not None:
        result["future_safety_distillation"] = provenance
    if counterfactual_provenance is not None:
        result["counterfactual_distillation"] = counterfactual_provenance
    return result


__all__ = ["deployment_checkpoint", "sha256_file"]
