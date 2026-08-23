#!/usr/bin/env python3
"""Select a PPO snapshot using repeatable multi-seed live evaluation."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_policy import evaluate


RUNTIME_DEFAULTS = {
    "hard_safety": False,
    "emergency_bomb_clearance": 0.0,
    "emergency_bomb_horizon": 6.0,
    "regular_bullet_safety_horizon": 0,
    "regular_bullet_safety_margin": 0.0,
    "regular_bullet_least_risk_fallback": False,
    "deathbomb_safety": False,
    "allow_bombs": True,
}


def runtime_config_mismatches(
    checkpoint_arguments: list[dict], overrides: dict
) -> dict[str, list]:
    """Find deployment settings that would differ across policy candidates."""
    mismatches = {}
    for name, default in RUNTIME_DEFAULTS.items():
        if overrides.get(name) is not None:
            continue
        values = [arguments.get(name, default) for arguments in checkpoint_arguments]
        if any(value != values[0] for value in values[1:]):
            mismatches[name] = values
    return mismatches


def _evaluate_task(task: tuple) -> tuple[str, dict]:
    """Evaluate one snapshot/seed pair in an isolated process."""
    (
        snapshot,
        image,
        deterministic,
        steps,
        seed,
        hard_safety,
        emergency_bomb_clearance,
        emergency_bomb_horizon,
        regular_bullet_safety_horizon,
        regular_bullet_safety_margin,
        regular_bullet_least_risk_fallback,
        deathbomb_safety,
        allow_bombs,
    ) = task
    result = evaluate(
        image=image,
        policy="checkpoint",
        checkpoint=snapshot,
        deterministic=deterministic,
        steps=steps,
        seed=seed,
        hard_safety=hard_safety,
        emergency_bomb_clearance=emergency_bomb_clearance,
        emergency_bomb_horizon=emergency_bomb_horizon,
        regular_bullet_safety_horizon=regular_bullet_safety_horizon,
        regular_bullet_safety_margin=regular_bullet_safety_margin,
        regular_bullet_least_risk_fallback=regular_bullet_least_risk_fallback,
        deathbomb_safety=deathbomb_safety,
        allow_bombs=allow_bombs,
    )
    return snapshot, result


def candidate_rank(candidate: dict) -> tuple[int, int, float]:
    """Order policies by NMNB completion, any completion, then stable return."""
    return (
        int(candidate.get("nmnb_successes", candidate["no_miss_successes"])),
        int(candidate["successes"]),
        float(candidate["selection_score"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("snapshots", nargs="+")
    parser.add_argument("--seeds", type=int, nargs="+", default=[41, 42, 43, 44])
    parser.add_argument("--steps", type=int, default=1_200)
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="parallel isolated emulator evaluations (default: 1)",
    )
    parser.add_argument("--lcb-z", type=float, default=1.0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--hard-safety", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--emergency-bomb-clearance", type=float, default=None)
    parser.add_argument("--emergency-bomb-horizon", type=float, default=None)
    parser.add_argument("--regular-bullet-safety-horizon", type=int, default=None)
    parser.add_argument("--regular-bullet-safety-margin", type=float, default=None)
    parser.add_argument(
        "--regular-bullet-least-risk-fallback",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--deathbomb-safety",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--allow-bombs",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--output", default="models/pc98_entity_ppo_best.pt")
    parser.add_argument("--report", default="runs/pc98rl/checkpoint_selection.json")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-runtime-config-comparison",
        action="store_true",
        help="allow an intentional safety/deployment configuration ablation",
    )
    args = parser.parse_args()

    snapshots = [Path(path).expanduser().resolve() for path in args.snapshots]
    missing = [str(path) for path in snapshots if not path.is_file()]
    if missing:
        parser.error("missing snapshot(s): " + ", ".join(missing))
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be unique")
    if args.jobs < 1:
        parser.error("jobs must be at least 1")

    import torch

    checkpoint_arguments = [
        torch.load(snapshot, map_location="cpu", weights_only=False).get("args", {})
        for snapshot in snapshots
    ]
    runtime_overrides = {
        "hard_safety": args.hard_safety,
        "emergency_bomb_clearance": args.emergency_bomb_clearance,
        "emergency_bomb_horizon": args.emergency_bomb_horizon,
        "regular_bullet_safety_horizon": args.regular_bullet_safety_horizon,
        "regular_bullet_safety_margin": args.regular_bullet_safety_margin,
        "regular_bullet_least_risk_fallback": (
            args.regular_bullet_least_risk_fallback
        ),
        "deathbomb_safety": args.deathbomb_safety,
        "allow_bombs": args.allow_bombs,
    }
    mismatches = runtime_config_mismatches(
        checkpoint_arguments, runtime_overrides
    )
    if mismatches and not args.allow_runtime_config_comparison:
        details = ", ".join(
            f"{name}={values}" for name, values in sorted(mismatches.items())
        )
        parser.error(
            "candidate runtime configurations differ; pass explicit overrides "
            "or --allow-runtime-config-comparison: " + details
        )

    snapshot_strings = [str(snapshot) for snapshot in snapshots]
    evaluations_by_snapshot: dict[str, list[dict]] = {
        snapshot: [] for snapshot in snapshot_strings
    }
    tasks = [
        (
            snapshot,
            args.image,
            args.deterministic,
            args.steps,
            seed,
            args.hard_safety,
            args.emergency_bomb_clearance,
            args.emergency_bomb_horizon,
            args.regular_bullet_safety_horizon,
            args.regular_bullet_safety_margin,
            args.regular_bullet_least_risk_fallback,
            args.deathbomb_safety,
            args.allow_bombs,
        )
        for snapshot in snapshot_strings
        for seed in args.seeds
    ]
    if args.jobs == 1:
        results = map(_evaluate_task, tasks)
        executor = None
    else:
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs)
        results = executor.map(_evaluate_task, tasks)
    try:
        for snapshot, result in results:
            evaluations_by_snapshot[snapshot].append(result)
            print(json.dumps({"snapshot": snapshot, **result}, sort_keys=True), flush=True)
    finally:
        if executor is not None:
            executor.shutdown(cancel_futures=True)

    candidates = []
    for snapshot in snapshots:
        evaluations = evaluations_by_snapshot[str(snapshot)]

        returns = np.asarray(
            [evaluation["scalar_return"] for evaluation in evaluations], dtype=np.float64
        )
        standard_error = float(returns.std(ddof=1) / math.sqrt(len(returns))) if len(returns) > 1 else 0.0
        candidate = {
            "snapshot": str(snapshot),
            "mean_return": round(float(returns.mean()), 6),
            "standard_error": round(standard_error, 6),
            "selection_score": round(float(returns.mean()) - args.lcb_z * standard_error, 6),
            "mean_deaths": round(
                float(np.mean([evaluation["death_events"] for evaluation in evaluations])), 4
            ),
            "successes": int(sum(evaluation["success"] for evaluation in evaluations)),
            "no_miss_successes": int(
                sum(evaluation["no_miss_success"] for evaluation in evaluations)
            ),
            "nmnb_successes": int(
                sum(evaluation.get("nmnb_success", False) for evaluation in evaluations)
            ),
            "evaluations": evaluations,
        }
        candidates.append(candidate)

    # An NMNB clear is the primary deployment objective. Ordinary completion
    # and lower-confidence-bound return only break ties between perfect clears.
    best = max(candidates, key=candidate_rank)
    report = {
        "seeds": args.seeds,
        "steps": args.steps,
        "jobs": args.jobs,
        "deterministic": args.deterministic,
        "hard_safety": args.hard_safety,
        "emergency_bomb_clearance": args.emergency_bomb_clearance,
        "emergency_bomb_horizon": args.emergency_bomb_horizon,
        "regular_bullet_safety_horizon": args.regular_bullet_safety_horizon,
        "regular_bullet_safety_margin": args.regular_bullet_safety_margin,
        "regular_bullet_least_risk_fallback": (
            args.regular_bullet_least_risk_fallback
        ),
        "deathbomb_safety": args.deathbomb_safety,
        "allow_bombs": args.allow_bombs,
        "allow_runtime_config_comparison": args.allow_runtime_config_comparison,
        "lcb_z": args.lcb_z,
        "selected": best["snapshot"],
        "candidates": candidates,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    if not args.dry_run:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best["snapshot"], output)
        print(f"selected {best['snapshot']} -> {output}", flush=True)
    else:
        print(f"selected {best['snapshot']} (dry run)", flush=True)


if __name__ == "__main__":
    main()
