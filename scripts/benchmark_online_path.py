#!/usr/bin/env python3
"""Benchmark the complete CPU decision path on a paused live TH05 state."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pc98rl.distributions import MaskedCategorical
from pc98rl.env import TH05CPUEnv, TH05_KINEMATICS
from pc98rl.latency import deadline_utilization, latency_summary
from pc98rl.model import EntityActorCritic
from pc98rl.safety import AuditedRegularBulletShield, DeathbombShield


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--iterations", type=int, default=2_000)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--deadline-ms", type=float, default=36.0)
    parser.add_argument("--regular-bullet-safety-horizon", type=int, default=6)
    parser.add_argument("--regular-bullet-safety-margin", type=float, default=0.0)
    parser.add_argument(
        "--regular-bullet-least-risk-fallback", action="store_true"
    )
    parser.add_argument(
        "--allow-bombs", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--report")
    args = parser.parse_args()
    if args.iterations < 1 or args.warmup < 0:
        parser.error("iterations must be positive and warmup must be non-negative")

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(args.seed)
    checkpoint_path = Path(args.checkpoint)
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    allow_bombs = (
        bool(saved.get("args", {}).get("allow_bombs", True))
        if args.allow_bombs is None
        else args.allow_bombs
    )
    deathbomb_safety = bool(
        allow_bombs and saved.get("args", {}).get("deathbomb_safety", False)
    )
    analytic_geometry = bool(saved.get("args", {}).get("analytic_geometry", False))
    model = EntityActorCritic(
        analytic_geometry=analytic_geometry,
        kinematic_spec=TH05_KINEMATICS if analytic_geometry else None,
    ).eval()
    model.load_state_dict(saved["model"])
    hidden = torch.zeros(1, 1, model.hidden_size)
    regular = AuditedRegularBulletShield(
        horizon_frames=args.regular_bullet_safety_horizon,
        extra_margin_px=args.regular_bullet_safety_margin,
        least_risk_fallback=args.regular_bullet_least_risk_fallback,
    )
    deathbomb = DeathbombShield() if deathbomb_safety else None
    env = TH05CPUEnv(args.image, deathbomb_guard=deathbomb_safety)
    shield_latencies: list[float] = []
    policy_latencies: list[float] = []
    total_latencies: list[float] = []
    try:
        observation, info = env.reset(seed=args.seed)
        observation_tensor = torch.from_numpy(observation).unsqueeze(0)
        raw_frame = info["raw_frame"]
        native_action_mask = info["action_mask"]
        with torch.no_grad():
            for iteration in range(args.warmup + args.iterations):
                total_started = time.perf_counter_ns()
                shield_started = total_started
                action_mask = native_action_mask.copy()
                if deathbomb is not None:
                    action_mask, _ = deathbomb.apply(raw_frame, action_mask)
                action_mask, _ = regular.apply(raw_frame, action_mask)
                if not allow_bombs:
                    action_mask[18] = False
                shield_finished = time.perf_counter_ns()
                logits, _, hidden = model.forward_step(observation_tensor, hidden)
                distribution = MaskedCategorical(
                    logits=logits,
                    valid_mask=torch.from_numpy(action_mask).unsqueeze(0),
                )
                distribution.sample()
                policy_finished = time.perf_counter_ns()
                if iteration >= args.warmup:
                    shield_latencies.append(
                        (shield_finished - shield_started) / 1_000_000.0
                    )
                    policy_latencies.append(
                        (policy_finished - shield_finished) / 1_000_000.0
                    )
                    total_latencies.append(
                        (policy_finished - total_started) / 1_000_000.0
                    )
    finally:
        env.close()

    total_summary = latency_summary(total_latencies)
    report = {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "image": str(Path(args.image).resolve()),
        "seed": args.seed,
        "torch_threads": torch.get_num_threads(),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "iterations": args.iterations,
        "warmup": args.warmup,
        "deadline_ms": args.deadline_ms,
        "shield": latency_summary(shield_latencies),
        "policy_and_sampling": latency_summary(policy_latencies),
        "complete_decision": total_summary,
        "p99_deadline_utilization": round(
            deadline_utilization(total_summary["p99_ms"], args.deadline_ms), 6
        ),
        "transactional_pause_during_inference": True,
        "regular_bullet_safety_horizon": args.regular_bullet_safety_horizon,
        "regular_bullet_least_risk_fallback": (
            args.regular_bullet_least_risk_fallback
        ),
        "allow_bombs": allow_bombs,
        "deathbomb_safety": deathbomb_safety,
    }
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
