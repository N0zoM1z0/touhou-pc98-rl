#!/usr/bin/env python3
"""Collect offline action-contrastive collision labels from exact TH05 states."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pc98rl.distributions import MaskedCategorical
from pc98rl.env import TH05CPUEnv, TH05_KINEMATICS
from pc98rl.model import EntityActorCritic
from pc98rl.safety import AuditedRegularBulletShield, DeathbombShield


def decode_survival_frames(raw_frame: Any, horizon: int) -> np.ndarray:
    values = raw_frame.regular_bullet_action_survival_frames(horizon, 0.0)
    if isinstance(values, (bytes, bytearray, memoryview)):
        return np.frombuffer(values, dtype=np.uint8).astype(np.int16)
    return np.asarray(values, dtype=np.int16)


def actor_step(
    model: EntityActorCritic,
    observation: np.ndarray,
    hidden: torch.Tensor,
    action_mask: np.ndarray,
) -> tuple[int, torch.Tensor, np.ndarray]:
    with torch.no_grad():
        logits, _, next_hidden = model.forward_step(
            torch.from_numpy(observation).unsqueeze(0), hidden
        )
        distribution = MaskedCategorical(
            logits=logits,
            valid_mask=torch.from_numpy(action_mask).unsqueeze(0),
        )
    return (
        int(distribution.mode.item()),
        next_hidden,
        logits.squeeze(0).numpy(force=True),
    )


def movement_mask(
    shield: AuditedRegularBulletShield,
    raw_frame: Any,
    structural_mask: np.ndarray,
) -> tuple[np.ndarray, int]:
    mask = np.asarray(structural_mask, dtype=np.bool_).copy()
    mask[18] = False
    shielded, _ = shield.apply(raw_frame, mask)
    unsafe = int(np.count_nonzero(mask[:18] & ~shielded[:18]))
    return shielded, unsafe


def branch_action(
    *,
    env: TH05CPUEnv,
    model: EntityActorCritic,
    shield: AuditedRegularBulletShield,
    anchor_hidden_after_observation: torch.Tensor,
    action: int,
    continuation_decisions: int,
    native_frames: int,
) -> dict[str, Any]:
    observation, info = env.load_branch_state()
    start_frame = int(info["raw_frame"].stage_frame())
    hidden = anchor_hidden_after_observation.clone()
    collision_decision = None
    collision_kind = None
    terminal = False
    end_flag = 0
    frames = []

    for decision in range(continuation_decisions + 1):
        if decision == 0:
            selected = action
        else:
            action_mask, _ = movement_mask(
                shield, info["raw_frame"], info["action_mask"]
            )
            selected, hidden, _ = actor_step(
                model, observation, hidden, action_mask
            )

        observation, _, terminal, truncated, info = env.step_native_frames(
            selected, native_frames=native_frames
        )
        current_frame = int(info["raw_frame"].stage_frame())
        frames.append(current_frame)
        pending_hit = bool(info["raw_frame"].deathbomb_window_active())
        miss = bool(info["miss_event"])
        if pending_hit or miss:
            collision_decision = decision + 1
            collision_kind = "miss" if miss else "pending_hit"
            break
        if terminal or truncated:
            break

    end_flag = int(info["end_flag"])
    return {
        "action": action,
        "collision_decision": collision_decision,
        "collision_native_frames": (
            None if collision_decision is None else collision_decision * native_frames
        ),
        "collision_kind": collision_kind,
        "terminal": bool(terminal),
        "end_flag": end_flag,
        "frames_evaluated": len(frames) * native_frames,
        "final_stage_frame": frames[-1] if frames else start_frame,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--search-decisions", type=int, default=1_500)
    parser.add_argument("--continuation-decisions", type=int, default=16)
    parser.add_argument("--native-frames", type=int, default=2)
    parser.add_argument("--safety-horizon", type=int, default=6)
    parser.add_argument("--min-unsafe-actions", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1601)
    args = parser.parse_args()

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(args.seed)
    checkpoint_path = Path(args.checkpoint).resolve()
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    analytic_geometry = bool(saved.get("args", {}).get("analytic_geometry", False))
    model = EntityActorCritic(
        analytic_geometry=analytic_geometry,
        kinematic_spec=TH05_KINEMATICS if analytic_geometry else None,
    ).eval()
    model.load_state_dict(saved["model"])
    hidden = torch.zeros(1, 1, model.hidden_size)

    shield = AuditedRegularBulletShield(horizon_frames=args.safety_horizon)
    deathbomb = DeathbombShield()
    env = TH05CPUEnv(
        args.image,
        deathbomb_guard=True,
        enable_state_branching=True,
    )
    report: dict[str, Any]
    try:
        observation, info = env.reset(seed=args.seed)
        anchor = None
        for search_decision in range(args.search_decisions):
            base_mask = np.asarray(info["action_mask"], dtype=np.bool_)
            runtime_mask, _ = deathbomb.apply(info["raw_frame"], base_mask)
            runtime_mask, unsafe_actions = movement_mask(
                shield, info["raw_frame"], runtime_mask
            )
            runtime_mask[18] = False
            behavior_action, next_hidden, _ = actor_step(
                model, observation, hidden, runtime_mask
            )

            if unsafe_actions >= args.min_unsafe_actions:
                anchor_observation, anchor_info = env.save_branch_state()
                anchor_mask, anchor_unsafe = movement_mask(
                    shield,
                    anchor_info["raw_frame"],
                    anchor_info["action_mask"],
                )
                if anchor_unsafe >= args.min_unsafe_actions:
                    behavior_action, anchor_next_hidden, behavior_logits = actor_step(
                        model, anchor_observation, hidden, anchor_mask
                    )
                    survival = decode_survival_frames(
                        anchor_info["raw_frame"], 16
                    )
                    outcomes = []
                    env.deathbomb_guard = False
                    for action in range(18):
                        if not bool(anchor_info["action_mask"][action]):
                            outcomes.append(
                                {
                                    "action": action,
                                    "collision_decision": None,
                                    "collision_native_frames": None,
                                    "collision_kind": "structurally_illegal",
                                    "terminal": False,
                                    "end_flag": 0,
                                    "frames_evaluated": 0,
                                    "final_stage_frame": int(
                                        anchor_info["raw_frame"].stage_frame()
                                    ),
                                }
                            )
                            continue
                        outcomes.append(
                            branch_action(
                                env=env,
                                model=model,
                                shield=shield,
                                anchor_hidden_after_observation=anchor_next_hidden,
                                action=action,
                                continuation_decisions=args.continuation_decisions,
                                native_frames=args.native_frames,
                            )
                        )
                    env.deathbomb_guard = True
                    env.load_branch_state()
                    anchor = {
                        "search_decision": search_decision,
                        "stage_frame": int(anchor_info["raw_frame"].stage_frame()),
                        "unsafe_actions_at_trigger_horizon": anchor_unsafe,
                        "h16_projected_survival_frames": survival[:18].tolist(),
                        "behavior_action": behavior_action,
                        "behavior_logits": behavior_logits.round(6).tolist(),
                        "observation_sha256": hashlib.sha256(
                            anchor_observation.tobytes()
                        ).hexdigest(),
                        "outcomes": outcomes,
                    }
                    break

            observation, _, terminal, truncated, info = env.step_native_frames(
                behavior_action, native_frames=args.native_frames
            )
            hidden = next_hidden
            if terminal or truncated:
                break

        if anchor is None:
            raise RuntimeError("no qualifying high-risk state was found")

        collisions = [
            outcome
            for outcome in anchor["outcomes"]
            if outcome["collision_decision"] is not None
        ]
        safe = [
            outcome
            for outcome in anchor["outcomes"]
            if outcome["collision_decision"] is None
            and outcome["collision_kind"] != "structurally_illegal"
        ]
        if collisions and safe:
            decision = (
                "The anchor yields action-dependent collision labels; proceed to a "
                "multi-anchor dataset and distilled actor experiment."
            )
        elif collisions:
            decision = (
                "Every legal action collides under this continuation; retain as a "
                "state-risk example but not as action-contrastive supervision."
            )
        else:
            decision = (
                "No legal action collides under the matched continuation. Reject "
                "this trigger as an action-risk label and seek a more immediate anchor."
            )
        report = {
            "research_question": (
                "Do exact branches from one audited high-risk TH05 state expose "
                "action-dependent collision outcomes for offline supervision?"
            ),
            "cpu_only": True,
            "online_modules_added": 0,
            "image": args.image,
            "checkpoint": args.checkpoint,
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "native_frames_per_decision": args.native_frames,
            "continuation_decisions": args.continuation_decisions,
            "safety_horizon": args.safety_horizon,
            "continuation": (
                f"deterministic actor with H{args.safety_horizon} "
                "regular-bullet shield and no bombs"
            ),
            "anchor": anchor,
            "summary": {
                "legal_actions": len(collisions) + len(safe),
                "collision_actions": len(collisions),
                "horizon_safe_actions": len(safe),
                "earliest_collision_native_frames": (
                    min(item["collision_native_frames"] for item in collisions)
                    if collisions
                    else None
                ),
            },
            "decision": decision,
        }
    finally:
        env.close()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
