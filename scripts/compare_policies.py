#!/usr/bin/env python3
"""Run a paired multi-seed checkpoint comparison in the live game."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_policy import evaluate


def _comparison_task(task: tuple) -> tuple[dict, dict]:
    (
        image,
        checkpoint,
        baseline_policy,
        baseline_checkpoint,
        deterministic,
        steps,
        seed,
        baseline_geometry,
        baseline_safety,
        allow_bombs,
    ) = task
    baseline = evaluate(
        image=image,
        policy=baseline_policy,
        checkpoint=baseline_checkpoint,
        deterministic=deterministic,
        steps=steps,
        seed=seed,
        analytic_geometry=baseline_geometry,
        hard_safety=baseline_safety,
        allow_bombs=allow_bombs,
    )
    checkpoint_result = evaluate(
        image=image,
        policy="checkpoint",
        checkpoint=checkpoint,
        deterministic=deterministic,
        steps=steps,
        seed=seed,
        allow_bombs=allow_bombs,
    )
    return baseline, checkpoint_result


def summarize(evaluations: list[dict]) -> dict:
    returns = np.asarray(
        [evaluation["scalar_return"] for evaluation in evaluations], dtype=np.float64
    )
    return {
        "mean_return": round(float(returns.mean()), 6),
        "return_standard_error": round(
            float(returns.std(ddof=1) / math.sqrt(len(returns)))
            if len(returns) > 1
            else 0.0,
            6,
        ),
        "deaths": int(sum(evaluation["death_events"] for evaluation in evaluations)),
        "successes": int(sum(evaluation["success"] for evaluation in evaluations)),
        "no_miss_successes": int(
            sum(evaluation.get("no_miss_success", False) for evaluation in evaluations)
        ),
        "nmnb_successes": int(
            sum(evaluation.get("nmnb_success", False) for evaluation in evaluations)
        ),
        "mean_raw_reward": np.mean(
            [evaluation["raw_reward"] for evaluation in evaluations], axis=0
        ).round(6).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", required=True)
    baseline = parser.add_mutually_exclusive_group()
    baseline.add_argument(
        "--baseline", choices=("random", "teacher", "untrained"), default="untrained"
    )
    baseline.add_argument(
        "--baseline-checkpoint",
        help="compare against another frozen checkpoint on the same seeds",
    )
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--steps", type=int, default=1_200)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--allow-bombs",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--report", default="runs/pc98rl/policy_comparison.json")
    args = parser.parse_args()

    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be unique")
    if args.jobs < 1:
        parser.error("jobs must be at least 1")

    baseline_policy = "checkpoint" if args.baseline_checkpoint else args.baseline
    baseline_geometry = False
    import torch

    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    target_safety = bool(saved.get("args", {}).get("hard_safety", False))
    # Frozen checkpoints retain their own adapter setting.  Non-checkpoint
    # baselines use the target's shield so the action-space constraint is matched.
    baseline_safety = None if baseline_policy == "checkpoint" else target_safety
    if baseline_policy == "untrained":
        baseline_geometry = bool(
            saved.get("args", {}).get("analytic_geometry", False)
        )

    baseline_results = []
    checkpoint_results = []
    tasks = [
        (
            args.image,
            args.checkpoint,
            baseline_policy,
            args.baseline_checkpoint,
            args.deterministic,
            args.steps,
            seed,
            baseline_geometry,
            baseline_safety,
            args.allow_bombs,
        )
        for seed in args.seeds
    ]
    if args.jobs == 1:
        results = map(_comparison_task, tasks)
        executor = None
    else:
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs)
        results = executor.map(_comparison_task, tasks)
    try:
        for baseline, checkpoint in results:
            baseline_results.append(baseline)
            checkpoint_results.append(checkpoint)
            print(json.dumps({"baseline": baseline, "checkpoint": checkpoint}, sort_keys=True))
    finally:
        if executor is not None:
            executor.shutdown(cancel_futures=True)

    baseline_summary = summarize(baseline_results)
    checkpoint_summary = summarize(checkpoint_results)
    paired_deltas = np.asarray(
        [
            checkpoint["scalar_return"] - baseline["scalar_return"]
            for baseline, checkpoint in zip(
                baseline_results, checkpoint_results, strict=True
            )
        ],
        dtype=np.float64,
    )
    report = {
        "seeds": args.seeds,
        "steps": args.steps,
        "jobs": args.jobs,
        "deterministic": args.deterministic,
        "allow_bombs": args.allow_bombs,
        "baseline_policy": baseline_policy,
        "baseline_checkpoint": (
            str(Path(args.baseline_checkpoint).resolve())
            if args.baseline_checkpoint
            else None
        ),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "baseline_summary": baseline_summary,
        "checkpoint_summary": checkpoint_summary,
        "paired_return_gain": round(float(paired_deltas.mean()), 6),
        "paired_return_gain_standard_error": round(
            float(paired_deltas.std(ddof=1) / math.sqrt(len(paired_deltas)))
            if len(paired_deltas) > 1
            else 0.0,
            6,
        ),
        "baseline_evaluations": baseline_results,
        "checkpoint_evaluations": checkpoint_results,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
