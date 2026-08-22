#!/usr/bin/env python3
"""Verify exact DOSBox-X save-state counterfactual replay on CPU."""

from __future__ import annotations

import argparse
import hashlib
import json
from typing import Any

import numpy as np

from pc98rl.env import TH05CPUEnv


def raw_digest(raw_frame: Any) -> str:
    return hashlib.sha256(bytes(raw_frame.__getstate__())).hexdigest()


def observation_digest(observation: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(observation, dtype=np.float32).tobytes()).hexdigest()


def rollout(
    env: TH05CPUEnv,
    *,
    action: int,
    decisions: int,
    native_frames: int,
) -> dict[str, Any]:
    miss_events = 0
    info: dict[str, Any] | None = None
    observation: np.ndarray | None = None
    frames = []
    for _ in range(decisions):
        observation, _, terminated, truncated, info = env.step_native_frames(
            action, native_frames=native_frames
        )
        frames.append(int(info["raw_frame"].stage_frame()))
        miss_events += int(info["miss_event"])
        if terminated or truncated:
            break
    assert observation is not None and info is not None
    return {
        "action": action,
        "decisions": len(frames),
        "stage_frames": frames,
        "miss_events": miss_events,
        "end_flag": int(info["end_flag"]),
        "observation_sha256": observation_digest(observation),
        "raw_frame_sha256": raw_digest(info["raw_frame"]),
        "observation": observation,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--action-a", type=int, default=10)
    parser.add_argument("--action-b", type=int, default=11)
    parser.add_argument("--decisions", type=int, default=8)
    parser.add_argument("--native-frames", type=int, default=2)
    parser.add_argument("--output")
    args = parser.parse_args()

    env = TH05CPUEnv(
        args.image,
        deathbomb_guard=True,
        enable_state_branching=True,
    )
    try:
        env.reset(seed=0)
        anchor_observation, anchor_info = env.save_branch_state()
        anchor = {
            "stage_frame": int(anchor_info["raw_frame"].stage_frame()),
            "observation_sha256": observation_digest(anchor_observation),
            "raw_frame_sha256": raw_digest(anchor_info["raw_frame"]),
        }

        first = rollout(
            env,
            action=args.action_a,
            decisions=args.decisions,
            native_frames=args.native_frames,
        )
        restored_observation, restored_info = env.load_branch_state()
        restored = {
            "stage_frame": int(restored_info["raw_frame"].stage_frame()),
            "observation_sha256": observation_digest(restored_observation),
            "raw_frame_sha256": raw_digest(restored_info["raw_frame"]),
        }
        repeat = rollout(
            env,
            action=args.action_a,
            decisions=args.decisions,
            native_frames=args.native_frames,
        )
        env.load_branch_state()
        contrast = rollout(
            env,
            action=args.action_b,
            decisions=args.decisions,
            native_frames=args.native_frames,
        )

        report = {
            "research_question": (
                "Can the offline collector replay action alternatives from an "
                "identical TH05 emulator state without changing the online actor?"
            ),
            "image": args.image,
            "cpu_only": True,
            "online_modules_added": 0,
            "method": {
                "backend": "private DOSBox-X save file controlled by exact PID/X11 window",
                "stage_frame_source": "MAIN.EXE _stage_frame",
                "stage_frame_player_relative_offset": int(
                    env._watcher.stage_frame_offset()
                ),
                "native_frames_per_decision": args.native_frames,
                "decisions_per_branch": args.decisions,
            },
            "anchor": anchor,
            "restored": restored,
            "anchor_exact": anchor == restored,
            "same_action_replay_exact": (
                first["observation_sha256"] == repeat["observation_sha256"]
                and first["raw_frame_sha256"] == repeat["raw_frame_sha256"]
                and first["stage_frames"] == repeat["stage_frames"]
            ),
            "contrasting_action_changes_state": (
                first["raw_frame_sha256"] != contrast["raw_frame_sha256"]
            ),
            "first": {key: value for key, value in first.items() if key != "observation"},
            "repeat": {
                key: value for key, value in repeat.items() if key != "observation"
            },
            "contrast": {
                key: value for key, value in contrast.items() if key != "observation"
            },
            "first_repeat_observation_max_abs_error": float(
                np.max(np.abs(first["observation"] - repeat["observation"]))
            ),
            "first_contrast_observation_max_abs_difference": float(
                np.max(np.abs(first["observation"] - contrast["observation"]))
            ),
            "decision": (
                "proceed to high-risk action-contrastive labeling; this probe "
                "establishes deterministic branching but does not yet establish "
                "a survival improvement"
            ),
            "limitations": [
                "The probe starts at an early safe pre-boss state.",
                "Collision-time labels and actor distillation are not evaluated yet.",
                "The X11/Xvfb controller is offline-only.",
            ],
        }
        rendered = json.dumps(report, indent=2, sort_keys=True)
        print(rendered)
        if args.output:
            from pathlib import Path

            Path(args.output).write_text(rendered + "\n", encoding="utf-8")
        if not report["anchor_exact"] or not report["same_action_replay_exact"]:
            raise SystemExit("save-state replay is not exact")
        if not report["contrasting_action_changes_state"]:
            raise SystemExit("contrasting actions did not change emulator state")
    finally:
        env.close()


if __name__ == "__main__":
    main()
