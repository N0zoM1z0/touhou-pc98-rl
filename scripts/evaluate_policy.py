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

from pc98rl.env import TH05CPUEnv
from pc98rl.heuristic import SafetyHeuristic


def evaluate(
    *,
    image: str,
    policy: str,
    checkpoint: str | None = None,
    deterministic: bool = False,
    steps: int = 1_200,
    seed: int = 20260822,
) -> dict:
    """Run one fixed-seed episode prefix and return JSON-serializable metrics."""
    rng = np.random.default_rng(seed)
    teacher = SafetyHeuristic()
    model = hidden = None
    if policy in ("untrained", "checkpoint"):
        import torch
        from torch.distributions import Categorical

        from pc98rl.model import EntityActorCritic

        torch.set_num_threads(1)
        torch.manual_seed(seed)
        model = EntityActorCritic().eval()
        if policy == "checkpoint":
            if checkpoint is None:
                raise ValueError("checkpoint policy requires a checkpoint path")
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            model.load_state_dict(saved["model"])
        hidden = torch.zeros(1, 1, model.hidden_size)
    env = TH05CPUEnv(image)
    reward_vector = np.zeros(3, dtype=np.float64)
    scalar_return = 0.0
    deaths = 0
    action_counts = np.zeros(19, dtype=np.int64)
    started = time.perf_counter()
    completed_steps = 0
    terminal = False
    end_flag = 0
    observation = np.zeros(env.observation_space.shape, dtype=np.float32)
    try:
        observation, _ = env.reset(seed=seed)
        for _ in range(steps):
            if policy == "random":
                action = int(rng.integers(19))
            elif policy == "teacher":
                action = teacher.act(observation)
            else:
                with torch.no_grad():
                    logits, _, hidden = model.forward_step(
                        torch.from_numpy(observation).unsqueeze(0), hidden
                    )
                    if deterministic:
                        action = int(logits.argmax(dim=-1).item())
                    else:
                        action = int(Categorical(logits=logits).sample().item())
            action_counts[action] += 1
            observation, reward, terminal, truncated, info = env.step(action)
            scalar_return += reward
            reward_vector += info["reward_vector"]
            deaths += int(info["reward_vector"][0] < -10.0)
            end_flag = int(info["end_flag"])
            completed_steps += 1
            if terminal or truncated:
                break
    finally:
        env.close()

    return {
        "policy": policy,
        "seed": seed,
        "steps": completed_steps,
        "elapsed_s": round(time.perf_counter() - started, 3),
        "scalar_return": round(scalar_return, 6),
        "raw_reward": reward_vector.round(4).tolist(),
        "death_events": deaths,
        "terminal": bool(terminal),
        "success": end_flag == 2,
        "end_flag": end_flag,
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
    )
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
