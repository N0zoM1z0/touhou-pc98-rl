#!/usr/bin/env python3
"""Select a PPO snapshot using repeatable multi-seed live evaluation."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_policy import evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("snapshots", nargs="+")
    parser.add_argument("--seeds", type=int, nargs="+", default=[41, 42, 43, 44])
    parser.add_argument("--steps", type=int, default=1_200)
    parser.add_argument("--lcb-z", type=float, default=1.0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--output", default="models/pc98_entity_ppo_best.pt")
    parser.add_argument("--report", default="runs/pc98rl/checkpoint_selection.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    snapshots = [Path(path).expanduser().resolve() for path in args.snapshots]
    missing = [str(path) for path in snapshots if not path.is_file()]
    if missing:
        parser.error("missing snapshot(s): " + ", ".join(missing))
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be unique")

    candidates = []
    for snapshot in snapshots:
        evaluations = []
        for seed in args.seeds:
            result = evaluate(
                image=args.image,
                policy="checkpoint",
                checkpoint=str(snapshot),
                deterministic=args.deterministic,
                steps=args.steps,
                seed=seed,
            )
            evaluations.append(result)
            print(json.dumps({"snapshot": str(snapshot), **result}, sort_keys=True), flush=True)

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
            "evaluations": evaluations,
        }
        candidates.append(candidate)

    # Terminal success is the primary objective; the lower-confidence-bound
    # return then favors policies that are both strong and stable across seeds.
    best = max(
        candidates,
        key=lambda candidate: (candidate["successes"], candidate["selection_score"]),
    )
    report = {
        "seeds": args.seeds,
        "steps": args.steps,
        "deterministic": args.deterministic,
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
