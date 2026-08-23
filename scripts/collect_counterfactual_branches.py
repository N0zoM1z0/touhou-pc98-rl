#!/usr/bin/env python3
"""Collect grouped, exact action-contrastive TH05 collision trajectories."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pc98rl.distributions import MaskedCategorical
from pc98rl.counterfactual import DATASET_FORMAT
from pc98rl.env import TH05CPUEnv, TH05_KINEMATICS
from pc98rl.model import EntityActorCritic
from pc98rl.safety import AuditedRegularBulletShield


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
    *,
    deterministic: bool = True,
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
        int((distribution.mode if deterministic else distribution.sample()).item()),
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
    continuation_shield: AuditedRegularBulletShield,
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
    frames = []

    for decision in range(continuation_decisions + 1):
        if decision == 0:
            selected = action
        else:
            action_mask, _ = movement_mask(
                continuation_shield, info["raw_frame"], info["action_mask"]
            )
            selected, hidden, _ = actor_step(
                model, observation, hidden, action_mask
            )

        observation, _, terminal, truncated, info = env.step_native_frames(
            selected, native_frames=native_frames
        )
        frames.append(int(info["raw_frame"].stage_frame()))
        pending_hit = bool(info["raw_frame"].deathbomb_window_active())
        miss = bool(info["miss_event"])
        if pending_hit or miss:
            collision_decision = decision + 1
            collision_kind = "miss" if miss else "pending_hit"
            break
        if terminal or truncated:
            break

    return {
        "action": action,
        "collision_decision": collision_decision,
        "collision_native_frames": (
            None if collision_decision is None else collision_decision * native_frames
        ),
        "collision_kind": collision_kind,
        "terminal": bool(terminal),
        "end_flag": int(info["end_flag"]),
        "frames_evaluated": len(frames) * native_frames,
        "final_stage_frame": frames[-1] if frames else start_frame,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _anchor_arrays(anchor: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    collision_frames = np.full(19, -2, dtype=np.int16)
    action_mask = np.zeros(19, dtype=np.bool_)
    for outcome in anchor["outcomes"]:
        action = int(outcome["action"])
        if outcome["collision_kind"] == "structurally_illegal":
            continue
        action_mask[action] = True
        collision_frames[action] = (
            -1
            if outcome["collision_native_frames"] is None
            else int(outcome["collision_native_frames"])
        )
    return action_mask, collision_frames


def _write_dataset(
    path: Path,
    *,
    anchors: list[dict[str, Any]],
    seed: int,
    checkpoint_sha256: str,
    trigger_horizon: int,
    continuation_horizon: int,
    continuation_decisions: int,
    native_frames: int,
    trajectory_policy: str,
) -> None:
    masks_and_frames = [_anchor_arrays(anchor) for anchor in anchors]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        format=np.asarray(DATASET_FORMAT),
        trajectory_id=np.asarray(f"seed_{seed}"),
        seed=np.asarray(seed, dtype=np.int64),
        checkpoint_sha256=np.asarray(checkpoint_sha256),
        actor_features=np.asarray(
            [anchor["_actor_feature"] for anchor in anchors], dtype=np.float32
        ),
        behavior_logits=np.asarray(
            [anchor["behavior_logits"] for anchor in anchors], dtype=np.float32
        ),
        action_masks=np.asarray([item[0] for item in masks_and_frames], dtype=np.bool_),
        collision_risk=np.asarray(
            [
                np.where(mask, frames >= 0, 0.0)
                for mask, frames in masks_and_frames
            ],
            dtype=np.float32,
        ),
        collision_native_frames=np.asarray(
            [item[1] for item in masks_and_frames], dtype=np.int16
        ),
        stage_frames=np.asarray(
            [anchor["stage_frame"] for anchor in anchors], dtype=np.int64
        ),
        search_decisions=np.asarray(
            [anchor["search_decision"] for anchor in anchors], dtype=np.int64
        ),
        behavior_actions=np.asarray(
            [anchor["behavior_action"] for anchor in anchors], dtype=np.int64
        ),
        trigger_unsafe_actions=np.asarray(
            [anchor["unsafe_actions_at_trigger_horizon"] for anchor in anchors],
            dtype=np.int16,
        ),
        projected_survival_frames=np.asarray(
            [anchor["projected_survival_frames"] for anchor in anchors],
            dtype=np.int16,
        ),
        trigger_horizon=np.asarray(trigger_horizon, dtype=np.int64),
        continuation_horizon=np.asarray(continuation_horizon, dtype=np.int64),
        continuation_decisions=np.asarray(continuation_decisions, dtype=np.int64),
        native_frames=np.asarray(native_frames, dtype=np.int64),
        trajectory_policy=np.asarray(trajectory_policy),
    )


def collect_trajectory(task: dict[str, Any]) -> dict[str, Any]:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    seed = int(task["seed"])
    torch.manual_seed(seed)
    checkpoint_path = Path(task["checkpoint"]).resolve()
    checkpoint_sha256 = sha256_file(checkpoint_path)
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    analytic_geometry = bool(saved.get("args", {}).get("analytic_geometry", False))
    model = EntityActorCritic(
        analytic_geometry=analytic_geometry,
        kinematic_spec=TH05_KINEMATICS if analytic_geometry else None,
    ).eval()
    model.load_state_dict(saved["model"])
    hidden = torch.zeros(1, 1, model.hidden_size)

    trigger_shield = AuditedRegularBulletShield(
        horizon_frames=int(task["trigger_horizon"])
    )
    continuation_shield = AuditedRegularBulletShield(
        horizon_frames=int(task["continuation_horizon"])
    )
    env = TH05CPUEnv(
        task["image"],
        deathbomb_guard=False,
        enable_state_branching=True,
    )
    anchors: list[dict[str, Any]] = []
    anchors_evaluated = 0
    rejected_noncontrastive = 0
    rejected_incomplete = 0
    terminal = False
    truncated = False
    last_anchor_decision = -int(task["min_anchor_gap"])
    completed_decisions = 0
    try:
        observation, info = env.reset(seed=seed)
        for search_decision in range(int(task["search_decisions"])):
            completed_decisions = search_decision + 1
            base_mask = np.asarray(info["action_mask"], dtype=np.bool_)
            runtime_mask, _ = movement_mask(
                continuation_shield, info["raw_frame"], base_mask
            )
            behavior_action, next_hidden, _ = actor_step(
                model,
                observation,
                hidden,
                runtime_mask,
                deterministic=task["trajectory_policy"] == "deterministic",
            )

            _, unsafe_actions = movement_mask(
                trigger_shield, info["raw_frame"], base_mask
            )
            anchor_due = (
                unsafe_actions >= int(task["min_unsafe_actions"])
                and search_decision - last_anchor_decision
                >= int(task["min_anchor_gap"])
            )
            if anchor_due:
                anchor_observation, anchor_info = env.save_branch_state()
                _, anchor_unsafe = movement_mask(
                    trigger_shield,
                    anchor_info["raw_frame"],
                    anchor_info["action_mask"],
                )
                if (
                    anchor_unsafe >= int(task["min_unsafe_actions"])
                    and not bool(anchor_info["raw_frame"].deathbomb_window_active())
                ):
                    anchor_runtime_mask, _ = movement_mask(
                        continuation_shield,
                        anchor_info["raw_frame"],
                        anchor_info["action_mask"],
                    )
                    behavior_action, anchor_next_hidden, behavior_logits = actor_step(
                        model,
                        anchor_observation,
                        hidden,
                        anchor_runtime_mask,
                        deterministic=task["trajectory_policy"] == "deterministic",
                    )
                    survival = decode_survival_frames(
                        anchor_info["raw_frame"],
                        max(16, int(task["continuation_horizon"])),
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
                        try:
                            outcome = branch_action(
                                env=env,
                                model=model,
                                continuation_shield=continuation_shield,
                                anchor_hidden_after_observation=anchor_next_hidden,
                                action=action,
                                continuation_decisions=int(
                                    task["continuation_decisions"]
                                ),
                                native_frames=int(task["native_frames"]),
                            )
                        except (RuntimeError, TimeoutError) as error:
                            outcome = {
                                "action": action,
                                "collision_decision": None,
                                "collision_native_frames": None,
                                "collision_kind": "branch_error",
                                "terminal": False,
                                "end_flag": 0,
                                "frames_evaluated": 0,
                                "final_stage_frame": int(
                                    anchor_info["raw_frame"].stage_frame()
                                ),
                                "error": f"{type(error).__name__}: {error}",
                            }
                        outcomes.append(outcome)
                    env.deathbomb_guard = True
                    observation, info = env.load_branch_state()
                    anchors_evaluated += 1
                    legal_outcomes = [
                        item
                        for item in outcomes
                        if item["collision_kind"] != "structurally_illegal"
                    ]
                    collisions = [
                        item
                        for item in legal_outcomes
                        if item["collision_decision"] is not None
                    ]
                    safe = [
                        item
                        for item in legal_outcomes
                        if item["collision_decision"] is None
                    ]
                    complete = not any(
                        item["collision_kind"] == "branch_error"
                        for item in legal_outcomes
                    )
                    contrastive = bool(complete and collisions and safe)
                    anchor = {
                        "search_decision": search_decision,
                        "stage_frame": int(anchor_info["raw_frame"].stage_frame()),
                        "unsafe_actions_at_trigger_horizon": anchor_unsafe,
                        "projected_survival_frames": survival[:18].tolist(),
                        "behavior_action": behavior_action,
                        "behavior_logits": behavior_logits.round(7).tolist(),
                        "observation_sha256": hashlib.sha256(
                            anchor_observation.tobytes()
                        ).hexdigest(),
                        "collision_actions": len(collisions),
                        "horizon_safe_actions": len(safe),
                        "contrastive": contrastive,
                        "complete": complete,
                        "outcomes": outcomes,
                        "_actor_feature": (
                            anchor_next_hidden.squeeze(0).squeeze(0).numpy(force=True)
                        ),
                    }
                    if not complete:
                        rejected_incomplete += 1
                    elif contrastive or not bool(task["require_action_contrast"]):
                        anchors.append(anchor)
                    else:
                        rejected_noncontrastive += 1
                    last_anchor_decision = search_decision
                    next_hidden = anchor_next_hidden

            observation, _, terminal, truncated, info = env.step_native_frames(
                behavior_action, native_frames=int(task["native_frames"])
            )
            hidden = next_hidden
            if terminal or truncated or len(anchors) >= int(task["max_anchors"]):
                break
    finally:
        env.close()

    if not anchors:
        raise RuntimeError(f"seed {seed}: no qualifying counterfactual anchor found")
    dataset_path = Path(task["dataset"])
    _write_dataset(
        dataset_path,
        anchors=anchors,
        seed=seed,
        checkpoint_sha256=checkpoint_sha256,
        trigger_horizon=int(task["trigger_horizon"]),
        continuation_horizon=int(task["continuation_horizon"]),
        continuation_decisions=int(task["continuation_decisions"]),
        native_frames=int(task["native_frames"]),
        trajectory_policy=str(task["trajectory_policy"]),
    )
    for anchor in anchors:
        anchor.pop("_actor_feature", None)
    collision_actions = sum(anchor["collision_actions"] for anchor in anchors)
    safe_actions = sum(anchor["horizon_safe_actions"] for anchor in anchors)
    report = {
        "format": DATASET_FORMAT,
        "cpu_only": True,
        "nmnb": True,
        "bomb_action_masked": True,
        "deathbomb_guard": False,
        "online_modules_added": 0,
        "image": task["image"],
        "checkpoint": task["checkpoint"],
        "checkpoint_sha256": checkpoint_sha256,
        "trajectory_id": f"seed_{seed}",
        "seed": seed,
        "native_frames_per_decision": int(task["native_frames"]),
        "trigger_horizon": int(task["trigger_horizon"]),
        "continuation_horizon": int(task["continuation_horizon"]),
        "continuation_decisions": int(task["continuation_decisions"]),
        "trajectory_policy": str(task["trajectory_policy"]),
        "continuation": (
            f"deterministic actor with H{int(task['continuation_horizon'])} "
            "regular-bullet shield and no bombs"
        ),
        "search_decisions_completed": completed_decisions,
        "terminal": bool(terminal),
        "truncated": bool(truncated),
        "anchors_evaluated": anchors_evaluated,
        "anchors_retained": len(anchors),
        "anchors_rejected_noncontrastive": rejected_noncontrastive,
        "anchors_rejected_incomplete": rejected_incomplete,
        "collision_actions": collision_actions,
        "horizon_safe_actions": safe_actions,
        "dataset": str(dataset_path),
        "anchors": anchors,
    }
    output = Path(task["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", help="legacy one-trajectory JSON output")
    parser.add_argument("--dataset", help="legacy one-trajectory NPZ output")
    parser.add_argument("--output-dir", help="multi-trajectory JSON/NPZ directory")
    parser.add_argument("--report", help="multi-trajectory aggregate JSON output")
    parser.add_argument("--search-decisions", type=int, default=1_500)
    parser.add_argument("--continuation-decisions", type=int, default=16)
    parser.add_argument("--native-frames", type=int, default=2)
    parser.add_argument("--trigger-horizon", type=int, default=2)
    parser.add_argument("--continuation-horizon", type=int, default=6)
    parser.add_argument(
        "--safety-horizon",
        type=int,
        help="legacy shorthand setting both trigger and continuation horizons",
    )
    parser.add_argument("--min-unsafe-actions", type=int, default=1)
    parser.add_argument("--max-anchors", type=int, default=4)
    parser.add_argument("--min-anchor-gap", type=int, default=24)
    parser.add_argument(
        "--trajectory-policy",
        choices=("deterministic", "sample"),
        default="deterministic",
        help="policy used only to navigate between exact branch anchors",
    )
    parser.add_argument(
        "--require-action-contrast",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed", type=int, default=1601)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse complete per-seed JSON/NPZ pairs after an interrupted collection",
    )
    args = parser.parse_args()

    if args.safety_horizon is not None:
        args.trigger_horizon = args.safety_horizon
        args.continuation_horizon = args.safety_horizon
    if args.trigger_horizon < 1 or args.continuation_horizon < 1:
        parser.error("trigger and continuation horizons must be positive")
    if args.max_anchors < 1 or args.min_anchor_gap < 1:
        parser.error("max anchors and minimum anchor gap must be positive")
    if args.jobs != 1:
        parser.error(
            "exact X11 save-state input requires jobs=1 per private DISPLAY"
        )

    seeds = args.seeds if args.seeds is not None else [args.seed]
    if len(set(seeds)) != len(seeds):
        parser.error("seeds must be unique")
    multi = args.seeds is not None
    if multi:
        if not args.output_dir or not args.report:
            parser.error("--seeds requires --output-dir and --report")
        if args.output or args.dataset:
            parser.error("use --output-dir/--report instead of --output/--dataset")
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if any(output_dir.iterdir()) and not args.resume:
            parser.error("output directory must be empty")
    else:
        if not args.output:
            parser.error("one-trajectory mode requires --output")
        output_dir = Path(args.output).parent

    tasks = []
    trajectories = []
    for seed in seeds:
        if multi:
            output = output_dir / f"seed_{seed}.json"
            dataset = output_dir / f"seed_{seed}.npz"
            if output.exists() or dataset.exists():
                if not args.resume or not output.is_file() or not dataset.is_file():
                    parser.error(f"incomplete or unexpected existing output for seed {seed}")
                trajectories.append(json.loads(output.read_text(encoding="utf-8")))
                continue
        else:
            output = Path(args.output)
            dataset = Path(args.dataset) if args.dataset else output.with_suffix(".npz")
        tasks.append(
            {
                "image": args.image,
                "checkpoint": args.checkpoint,
                "output": str(output),
                "dataset": str(dataset),
                "search_decisions": args.search_decisions,
                "continuation_decisions": args.continuation_decisions,
                "native_frames": args.native_frames,
                "trigger_horizon": args.trigger_horizon,
                "continuation_horizon": args.continuation_horizon,
                "min_unsafe_actions": args.min_unsafe_actions,
                "max_anchors": args.max_anchors,
                "min_anchor_gap": args.min_anchor_gap,
                "require_action_contrast": args.require_action_contrast,
                "trajectory_policy": args.trajectory_policy,
                "seed": seed,
            }
        )

    if args.jobs == 1:
        results = map(collect_trajectory, tasks)
        executor = None
    else:
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs)
        results = executor.map(collect_trajectory, tasks)
    try:
        for result in results:
            trajectories.append(result)
            print(
                json.dumps(
                    {
                        key: result[key]
                        for key in (
                            "trajectory_id",
                            "anchors_retained",
                            "collision_actions",
                            "horizon_safe_actions",
                        )
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        if executor is not None:
            executor.shutdown(cancel_futures=True)

    if multi:
        aggregate = {
            "research_question": (
                "Do disjoint exact H2-triggered trajectories provide repeated "
                "action-contrastive collision labels under the deployed H6 continuation?"
            ),
            "format": DATASET_FORMAT,
            "cpu_only": True,
            "nmnb": True,
            "bomb_action_masked": True,
            "deathbomb_guard": False,
            "online_modules_added": 0,
            "image": args.image,
            "checkpoint": args.checkpoint,
            "seeds": seeds,
            "jobs": args.jobs,
            "trigger_horizon": args.trigger_horizon,
            "continuation_horizon": args.continuation_horizon,
            "max_anchors_per_trajectory": args.max_anchors,
            "require_action_contrast": args.require_action_contrast,
            "trajectory_policy": args.trajectory_policy,
            "trajectory_count": len(trajectories),
            "anchor_count": sum(item["anchors_retained"] for item in trajectories),
            "collision_actions": sum(item["collision_actions"] for item in trajectories),
            "horizon_safe_actions": sum(
                item["horizon_safe_actions"] for item in trajectories
            ),
            "trajectories": [
                {
                    "trajectory_id": item["trajectory_id"],
                    "seed": item["seed"],
                    "anchors_evaluated": item["anchors_evaluated"],
                    "anchors_retained": item["anchors_retained"],
                    "anchors_rejected_noncontrastive": item.get(
                        "anchors_rejected_noncontrastive", 0
                    ),
                    "anchors_rejected_incomplete": item.get(
                        "anchors_rejected_incomplete", 0
                    ),
                    "collision_actions": item["collision_actions"],
                    "horizon_safe_actions": item["horizon_safe_actions"],
                    "dataset": item["dataset"],
                }
                for item in trajectories
            ],
        }
        report = Path(args.report)
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n")
        print(json.dumps(aggregate, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
