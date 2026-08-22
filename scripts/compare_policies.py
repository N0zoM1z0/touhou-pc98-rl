#!/usr/bin/env python3
"""Run a paired multi-seed checkpoint comparison in the live game."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_policy import evaluate


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
        "mean_raw_reward": np.mean(
            [evaluation["raw_reward"] for evaluation in evaluations], axis=0
        ).round(6).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--baseline", choices=("random", "teacher", "untrained"), default="untrained"
    )
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--steps", type=int, default=1_200)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--report", default="runs/pc98rl/policy_comparison.json")
    args = parser.parse_args()

    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be unique")

    baseline_results = []
    checkpoint_results = []
    for seed in args.seeds:
        baseline = evaluate(
            image=args.image,
            policy=args.baseline,
            deterministic=args.deterministic,
            steps=args.steps,
            seed=seed,
        )
        checkpoint = evaluate(
            image=args.image,
            policy="checkpoint",
            checkpoint=args.checkpoint,
            deterministic=args.deterministic,
            steps=args.steps,
            seed=seed,
        )
        baseline_results.append(baseline)
        checkpoint_results.append(checkpoint)
        print(json.dumps({"baseline": baseline, "checkpoint": checkpoint}, sort_keys=True))

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
        "deterministic": args.deterministic,
        "baseline_policy": args.baseline,
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
