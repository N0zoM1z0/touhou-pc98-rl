#!/usr/bin/env python3
"""Collect complete no-miss TH05 trajectories for offline distillation."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _collect(task: tuple) -> dict:
    (
        image,
        checkpoint,
        output_dir,
        seed,
        steps,
        regular_horizon,
        regular_margin,
    ) = task
    import torch

    from pc98rl.distributions import MaskedCategorical
    from pc98rl.env import TH05CPUEnv, TH05_KINEMATICS
    from pc98rl.model import EntityActorCritic
    from pc98rl.safety import AuditedRegularBulletShield, DeathbombShield

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(seed)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    analytic_geometry = bool(saved.get("args", {}).get("analytic_geometry", False))
    model = EntityActorCritic(
        analytic_geometry=analytic_geometry,
        kinematic_spec=TH05_KINEMATICS if analytic_geometry else None,
    ).eval()
    model.load_state_dict(saved["model"])
    hidden = torch.zeros(1, 1, model.hidden_size)
    regular = AuditedRegularBulletShield(
        horizon_frames=regular_horizon,
        extra_margin_px=regular_margin,
    )
    deathbomb = DeathbombShield()
    env = TH05CPUEnv(image, deathbomb_guard=True)
    observations: list[np.ndarray] = []
    actions: list[int] = []
    masks: list[np.ndarray] = []
    policy_valid: list[bool] = []
    deaths = 0
    bombs_used = 0
    success = False
    end_flag = 0
    try:
        observation, info = env.reset(seed=seed)
        previous_bombs = int(round(float(observation[10]) * 8.0))
        for step in range(steps):
            action_mask, _ = deathbomb.apply(info["raw_frame"], info["action_mask"])
            action_mask, _ = regular.apply(info["raw_frame"], action_mask)
            with torch.no_grad():
                logits, _, hidden = model.forward_step(
                    torch.from_numpy(observation).unsqueeze(0), hidden
                )
                distribution = MaskedCategorical(
                    logits=logits,
                    valid_mask=torch.from_numpy(action_mask).unsqueeze(0),
                )
                action = int(distribution.sample().item())

            observations.append(observation.copy())
            actions.append(action)
            masks.append(action_mask.copy())
            next_observation, _, terminal, truncated, next_info = env.step(action)
            online_override = bool(next_info.get("deathbomb_intervention", False))
            policy_valid.append(not (online_override and action != 18) and action != 18)
            current_bombs = int(round(float(next_observation[10]) * 8.0))
            bombs_used += max(previous_bombs - current_bombs, 0)
            previous_bombs = current_bombs
            deaths += int(next_info["miss_event"])
            observation = next_observation
            info = next_info
            end_flag = int(info["end_flag"])
            if terminal or truncated:
                success = end_flag == 2
                break
    finally:
        env.close()

    completed_steps = len(actions)
    elite = bool(success and deaths == 0)
    trajectory_path = None
    if elite:
        trajectory_path = Path(output_dir) / f"seed_{seed}.npz"
        np.savez_compressed(
            trajectory_path,
            observations=np.asarray(observations, dtype=np.float32),
            actions=np.asarray(actions, dtype=np.int64),
            action_masks=np.asarray(masks, dtype=np.bool_),
            policy_valid=np.asarray(policy_valid, dtype=np.bool_),
            bombs_used=np.asarray(bombs_used, dtype=np.int64),
            seed=np.asarray(seed, dtype=np.int64),
        )
    return {
        "seed": seed,
        "steps": completed_steps,
        "success": success,
        "no_miss_success": elite,
        "deaths": deaths,
        "bombs_used": bombs_used,
        "attributable_actions": int(np.count_nonzero(policy_valid)),
        "trajectory": str(trajectory_path) if trajectory_path else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--steps", type=int, default=2400)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--regular-bullet-safety-horizon", type=int, default=6)
    parser.add_argument("--regular-bullet-safety-margin", type=float, default=0.0)
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be unique")
    if args.jobs < 1:
        parser.error("jobs must be positive")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = list(output_dir.glob("seed_*.npz"))
    if existing:
        parser.error(f"output directory already contains {len(existing)} trajectories")

    tasks = [
        (
            args.image,
            args.checkpoint,
            str(output_dir),
            seed,
            args.steps,
            args.regular_bullet_safety_horizon,
            args.regular_bullet_safety_margin,
        )
        for seed in args.seeds
    ]
    if args.jobs == 1:
        results = map(_collect, tasks)
        executor = None
    else:
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs)
        results = executor.map(_collect, tasks)
    episodes = []
    try:
        for result in results:
            episodes.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)
    finally:
        if executor is not None:
            executor.shutdown(cancel_futures=True)
    report = {
        "image": str(Path(args.image).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "seeds": args.seeds,
        "steps": args.steps,
        "regular_bullet_safety_horizon": args.regular_bullet_safety_horizon,
        "regular_bullet_safety_margin": args.regular_bullet_safety_margin,
        "deathbomb_safety": True,
        "attempts": len(episodes),
        "elites": int(sum(episode["no_miss_success"] for episode in episodes)),
        "elite_transitions": int(
            sum(episode["attributable_actions"] for episode in episodes if episode["no_miss_success"])
        ),
        "episodes": episodes,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in ("attempts", "elites", "elite_transitions")}, sort_keys=True))


if __name__ == "__main__":
    main()
