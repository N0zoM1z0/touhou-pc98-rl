#!/usr/bin/env python3
"""Evaluate a policy in the live TH05 environment."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pc98rl.distributions import MaskedCategorical
from pc98rl.env import TH05CPUEnv, TH05_KINEMATICS, describe_th05_scenario
from pc98rl.heuristic import SafetyHeuristic
from pc98rl.safety import (
    AuditedRegularBulletShield,
    DeathbombShield,
    EmergencyBombShield,
)


def evaluate(
    *,
    image: str,
    policy: str,
    checkpoint: str | None = None,
    deterministic: bool = False,
    steps: int = 1_200,
    seed: int = 20260822,
    analytic_geometry: bool = False,
    hard_safety: bool | None = None,
    emergency_bomb_clearance: float | None = None,
    emergency_bomb_horizon: float | None = None,
    regular_bullet_safety_horizon: int | None = None,
    regular_bullet_safety_margin: float | None = None,
    deathbomb_safety: bool | None = None,
) -> dict:
    """Run one fixed-seed episode prefix and return JSON-serializable metrics."""
    rng = np.random.default_rng(seed)
    teacher = SafetyHeuristic()
    model = hidden = None
    if policy in ("untrained", "checkpoint"):
        import torch
        from pc98rl.model import EntityActorCritic

        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        saved = None
        if policy == "checkpoint":
            if checkpoint is None:
                raise ValueError("checkpoint policy requires a checkpoint path")
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            analytic_geometry = bool(
                saved.get("args", {}).get("analytic_geometry", False)
            )
            if hard_safety is None:
                hard_safety = bool(saved.get("args", {}).get("hard_safety", False))
            if emergency_bomb_clearance is None:
                emergency_bomb_clearance = float(
                    saved.get("args", {}).get("emergency_bomb_clearance", 0.0)
                )
            if emergency_bomb_horizon is None:
                emergency_bomb_horizon = float(
                    saved.get("args", {}).get("emergency_bomb_horizon", 6.0)
                )
            if regular_bullet_safety_horizon is None:
                regular_bullet_safety_horizon = int(
                    saved.get("args", {}).get("regular_bullet_safety_horizon", 0)
                )
            if regular_bullet_safety_margin is None:
                regular_bullet_safety_margin = float(
                    saved.get("args", {}).get("regular_bullet_safety_margin", 0.0)
                )
            if deathbomb_safety is None:
                deathbomb_safety = bool(
                    saved.get("args", {}).get("deathbomb_safety", False)
                )
        torch.manual_seed(seed)
        model = EntityActorCritic(
            analytic_geometry=analytic_geometry,
            kinematic_spec=TH05_KINEMATICS if analytic_geometry else None,
        ).eval()
        if saved is not None:
            model.load_state_dict(saved["model"])
        hidden = torch.zeros(1, 1, model.hidden_size)
    if hard_safety is None:
        hard_safety = False
    if emergency_bomb_clearance is None:
        emergency_bomb_clearance = 0.0
    if emergency_bomb_horizon is None:
        emergency_bomb_horizon = 6.0
    if regular_bullet_safety_horizon is None:
        regular_bullet_safety_horizon = 0
    if regular_bullet_safety_margin is None:
        regular_bullet_safety_margin = 0.0
    if deathbomb_safety is None:
        deathbomb_safety = False
    regular_bullet_shield = (
        AuditedRegularBulletShield(
            horizon_frames=regular_bullet_safety_horizon,
            extra_margin_px=regular_bullet_safety_margin,
        )
        if regular_bullet_safety_horizon > 0
        else None
    )
    deathbomb_shield = DeathbombShield() if deathbomb_safety else None
    bomb_shield = (
        EmergencyBombShield(
            TH05_KINEMATICS,
            clearance_px=emergency_bomb_clearance,
            horizon_steps=emergency_bomb_horizon,
        )
        if emergency_bomb_clearance > 0.0
        else None
    )
    env = TH05CPUEnv(image, deathbomb_guard=deathbomb_safety)
    reward_vector = np.zeros(3, dtype=np.float64)
    scalar_return = 0.0
    deaths = 0
    action_counts = np.zeros(19, dtype=np.int64)
    started = time.perf_counter()
    completed_steps = 0
    terminal = False
    end_flag = 0
    constrained_steps = 0
    removed_probability_mass = 0.0
    emergency_bomb_interventions = 0
    regular_bullet_interventions = 0
    deathbomb_interventions = 0
    observation = np.zeros(env.observation_space.shape, dtype=np.float32)
    scenario = None
    try:
        observation, info = env.reset(seed=seed)
        scenario = describe_th05_scenario(observation)
        action_mask = info["action_mask"]
        raw_frame = info["raw_frame"]
        for _ in range(steps):
            effective_action_mask = (
                action_mask.copy()
                if hard_safety
                or regular_bullet_shield is not None
                or deathbomb_shield is not None
                else np.ones_like(action_mask, dtype=np.bool_)
            )
            if deathbomb_shield is not None:
                effective_action_mask, deathbomb_intervention = deathbomb_shield.apply(
                    raw_frame, effective_action_mask
                )
            else:
                deathbomb_intervention = False
            if regular_bullet_shield is not None:
                effective_action_mask, intervention = regular_bullet_shield.apply(
                    raw_frame, effective_action_mask
                )
                regular_bullet_interventions += int(intervention)
            intervention = False
            if bomb_shield is not None:
                effective_action_mask, intervention = bomb_shield.apply(
                    observation, effective_action_mask
                )
                emergency_bomb_interventions += int(intervention)
            if policy == "random":
                if (
                    hard_safety
                    or bomb_shield is not None
                    or regular_bullet_shield is not None
                    or deathbomb_shield is not None
                ):
                    valid_actions = np.flatnonzero(effective_action_mask)
                    action = int(rng.choice(valid_actions))
                    removed_probability_mass += 1.0 - len(valid_actions) / 19.0
                else:
                    action = int(rng.integers(19))
            elif policy == "teacher":
                action = teacher.act(observation)
                if not effective_action_mask[action]:
                    action = int(np.flatnonzero(effective_action_mask)[0])
            else:
                with torch.no_grad():
                    logits, _, hidden = model.forward_step(
                        torch.from_numpy(observation).unsqueeze(0), hidden
                    )
                    distribution = MaskedCategorical(
                        logits=logits,
                        valid_mask=(
                            torch.from_numpy(effective_action_mask).unsqueeze(0)
                            if hard_safety
                            or bomb_shield is not None
                            or regular_bullet_shield is not None
                            or deathbomb_shield is not None
                            else None
                        ),
                    )
                    removed_probability_mass += float(
                        distribution.removed_probability_mass.item()
                    )
                    if deterministic:
                        action = int(distribution.mode.item())
                    else:
                        action = int(distribution.sample().item())
            constrained_steps += int(np.any(~effective_action_mask))
            action_counts[action] += 1
            observation, reward, terminal, truncated, info = env.step(action)
            deathbomb_interventions += int(
                deathbomb_intervention
                or info.get("deathbomb_intervention", False)
            )
            action_mask = info["action_mask"]
            raw_frame = info["raw_frame"]
            scalar_return += reward
            reward_vector += info["reward_vector"]
            deaths += int(info["miss_event"])
            end_flag = int(info["end_flag"])
            completed_steps += 1
            if terminal or truncated:
                break
    finally:
        env.close()

    success = end_flag == 2
    return {
        "policy": policy,
        "seed": seed,
        "steps": completed_steps,
        "elapsed_s": round(time.perf_counter() - started, 3),
        "scalar_return": round(scalar_return, 6),
        "raw_reward": reward_vector.round(4).tolist(),
        "death_events": deaths,
        "terminal": bool(terminal),
        "success": success,
        "no_miss_success": success and deaths == 0,
        "end_flag": end_flag,
        "scenario": scenario,
        "analytic_geometry": analytic_geometry if model is not None else None,
        "hard_safety": hard_safety,
        "emergency_bomb_clearance": emergency_bomb_clearance,
        "emergency_bomb_horizon": emergency_bomb_horizon,
        "emergency_bomb_interventions": emergency_bomb_interventions,
        "regular_bullet_safety_horizon": regular_bullet_safety_horizon,
        "regular_bullet_safety_margin": regular_bullet_safety_margin,
        "regular_bullet_interventions": regular_bullet_interventions,
        "deathbomb_safety": deathbomb_safety,
        "deathbomb_interventions": deathbomb_interventions,
        "constrained_step_fraction": round(
            constrained_steps / completed_steps if completed_steps else 0.0, 6
        ),
        "removed_probability_mass": round(
            removed_probability_mass / completed_steps if completed_steps else 0.0,
            6,
        ),
        "final_xy": observation[:2].round(4).tolist(),
        "action_counts": action_counts.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument(
        "--policy", choices=("random", "teacher", "untrained", "checkpoint"), required=True
    )
    parser.add_argument("--checkpoint", default="models/pc98_entity_ppo.pt")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--analytic-geometry", action="store_true")
    parser.add_argument(
        "--hard-safety",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="override the checkpoint's adapter-certified action mask setting",
    )
    parser.add_argument("--emergency-bomb-clearance", type=float, default=None)
    parser.add_argument("--emergency-bomb-horizon", type=float, default=None)
    parser.add_argument("--regular-bullet-safety-horizon", type=int, default=None)
    parser.add_argument("--regular-bullet-safety-margin", type=float, default=None)
    parser.add_argument(
        "--deathbomb-safety",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--steps", type=int, default=1_200)
    parser.add_argument("--seed", type=int, default=20260822)
    args = parser.parse_args()
    result = evaluate(
        image=args.image,
        policy=args.policy,
        checkpoint=args.checkpoint,
        deterministic=args.deterministic,
        steps=args.steps,
        seed=args.seed,
        analytic_geometry=args.analytic_geometry,
        hard_safety=args.hard_safety,
        emergency_bomb_clearance=args.emergency_bomb_clearance,
        emergency_bomb_horizon=args.emergency_bomb_horizon,
        regular_bullet_safety_horizon=args.regular_bullet_safety_horizon,
        regular_bullet_safety_margin=args.regular_bullet_safety_margin,
        deathbomb_safety=args.deathbomb_safety,
    )
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
