#!/usr/bin/env python3
"""Export a compact actor checkpoint and strip all offline training state."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pc98rl.checkpoints import deployment_checkpoint, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--regular-bullet-safety-horizon", type=int)
    parser.add_argument(
        "--regular-bullet-least-risk-fallback",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    args = parser.parse_args()
    source = Path(args.checkpoint)
    output = Path(args.output)
    if not source.is_file():
        parser.error(f"checkpoint does not exist: {source}")
    if source.resolve() == output.resolve():
        parser.error("output must differ from the training checkpoint")
    if args.regular_bullet_safety_horizon is not None and not (
        1 <= args.regular_bullet_safety_horizon <= 16
    ):
        parser.error("regular-bullet-safety-horizon must be between 1 and 16")

    overrides = {}
    if args.regular_bullet_safety_horizon is not None:
        overrides["regular_bullet_safety_horizon"] = (
            args.regular_bullet_safety_horizon
        )
    if args.regular_bullet_least_risk_fallback is not None:
        overrides["regular_bullet_least_risk_fallback"] = (
            args.regular_bullet_least_risk_fallback
        )
    saved = torch.load(source, map_location="cpu", weights_only=False)
    exported = deployment_checkpoint(
        saved, source=source, argument_overrides=overrides
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(exported, output)
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "sha256": sha256_file(output),
                "bytes": output.stat().st_size,
                "source_bytes": source.stat().st_size,
                "keys": sorted(exported),
                "runtime_overrides": overrides,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
