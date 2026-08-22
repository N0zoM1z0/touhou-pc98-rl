"""Synchronous recurrent PPO tuned for CPU-only TH05 training."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import time
import traceback
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch
from torch import nn

from .distributions import MaskedCategorical
from .env import (
    DEFAULT_REWARD_WEIGHTS,
    TH05CPUEnv,
    TH05_KINEMATICS,
    describe_th05_scenario,
)
from .model import FEATURE_DIM, EntityActorCritic


def _worker_main(connection, image: str, frame_interval_s: float, seed: int) -> None:
    """Own one DOSBox-X process and keep all emulator work off the trainer."""
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    env = TH05CPUEnv(image, frame_interval_s=frame_interval_s)
    try:
        observation, reset_info = env.reset(seed=seed)
        episode_deaths = 0
        connection.send(("ready", observation, reset_info["action_mask"]))
        while True:
            command, payload = connection.recv()
            if command == "close":
                return
            if command != "step":
                raise ValueError(f"unknown worker command {command}")

            observation, _, terminated, truncated, info = env.step(int(payload))
            done = bool(terminated or truncated)
            terminal_flag = int(info["end_flag"])
            scaled_reward = info["scaled_reward_vector"]
            episode_deaths += int(scaled_reward[0] < -0.1)
            no_miss_success = bool(
                done and terminal_flag == 2 and episode_deaths == 0
            )
            if done:
                observation, reset_info = env.reset()
                action_mask = reset_info["action_mask"]
                episode_deaths = 0
            else:
                action_mask = info["action_mask"]
            connection.send(
                (
                    "step",
                    observation,
                    scaled_reward,
                    done,
                    terminal_flag,
                    action_mask,
                    no_miss_success,
                )
            )
    except BaseException:
        connection.send(("error", traceback.format_exc()))
    finally:
        env.close()
        connection.close()


class WorkerPool:
    def __init__(
        self,
        image: str,
        workers: int,
        frame_interval_s: float,
        seed: int,
        timeout_s: float,
    ):
        context = mp.get_context("spawn")
        self.timeout_s = timeout_s
        self.processes = []
        self.connections = []
        for worker_id in range(workers):
            parent, child = context.Pipe()
            process = context.Process(
                target=_worker_main,
                args=(child, image, frame_interval_s, seed + worker_id),
                daemon=True,
            )
            process.start()
            child.close()
            self.processes.append(process)
            self.connections.append(parent)

        observations = []
        action_masks = []
        try:
            for worker_id, connection in enumerate(self.connections):
                if not connection.poll(self.timeout_s):
                    raise TimeoutError(
                        f"worker {worker_id} did not start within {self.timeout_s}s"
                    )
                message = connection.recv()
                if message[0] == "error":
                    raise RuntimeError(f"worker {worker_id} failed:\n{message[1]}")
                observations.append(message[1])
                action_masks.append(message[2])
        except BaseException:
            self.close()
            raise
        self.observations = np.stack(observations).astype(np.float32, copy=False)
        self.action_masks = np.stack(action_masks).astype(np.bool_, copy=False)

    def step(self, actions: np.ndarray):
        for connection, action in zip(self.connections, actions, strict=True):
            connection.send(("step", int(action)))
        observations, rewards, dones, flags, action_masks, no_miss_successes = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for worker_id, connection in enumerate(self.connections):
            if not connection.poll(self.timeout_s):
                raise TimeoutError(
                    f"worker {worker_id} did not step within {self.timeout_s}s"
                )
            message = connection.recv()
            if message[0] == "error":
                raise RuntimeError(f"worker {worker_id} failed:\n{message[1]}")
            _, observation, reward, done, flag, action_mask, no_miss_success = message
            observations.append(observation)
            rewards.append(reward)
            dones.append(done)
            flags.append(flag)
            action_masks.append(action_mask)
            no_miss_successes.append(no_miss_success)
        self.observations = np.stack(observations).astype(np.float32, copy=False)
        self.action_masks = np.stack(action_masks).astype(np.bool_, copy=False)
        return (
            self.observations,
            np.stack(rewards).astype(np.float32, copy=False),
            np.asarray(dones, dtype=np.float32),
            np.asarray(flags, dtype=np.uint8),
            self.action_masks,
            np.asarray(no_miss_successes, dtype=np.bool_),
        )

    def close(self) -> None:
        for connection, process in zip(self.connections, self.processes, strict=True):
            if process.is_alive():
                try:
                    connection.send(("close", None))
                except (BrokenPipeError, EOFError):
                    pass
        for connection, process in zip(self.connections, self.processes, strict=True):
            process.join(timeout=3.0)
            if process.is_alive():
                process.kill()
                process.join()
            connection.close()


def _vector_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_values: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards)
    accumulator = np.zeros_like(last_values)
    for step in range(len(rewards) - 1, -1, -1):
        next_values = last_values if step == len(rewards) - 1 else values[step + 1]
        not_done = (1.0 - dones[step])[:, None]
        delta = rewards[step] + gamma * next_values * not_done - values[step]
        accumulator = delta + gamma * gae_lambda * not_done * accumulator
        advantages[step] = accumulator
    return advantages, advantages + values


def _as_sequences(array: np.ndarray, sequence_length: int) -> np.ndarray:
    # [time, env, ...] -> [env * time/sequence, sequence, ...]
    env_first = np.swapaxes(array, 0, 1)
    return env_first.reshape(
        env_first.shape[0] * env_first.shape[1] // sequence_length,
        sequence_length,
        *env_first.shape[2:],
    )


def _apply_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint: dict,
    *,
    resume: bool,
) -> tuple[int, int]:
    """Load policy weights, optionally restoring same-run optimizer state."""
    model.load_state_dict(checkpoint["model"])
    if not resume:
        return 0, 0
    optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint["update"]), int(checkpoint["environment_steps"])


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def train(args: argparse.Namespace) -> None:
    if args.rollout_steps % args.sequence_length:
        raise ValueError("rollout-steps must be divisible by sequence-length")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(args.metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    first_update = 0
    environment_steps = 0
    resume_checkpoint = None
    initialization_checkpoint = None
    initialization = None
    if args.resume:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"cannot resume because checkpoint does not exist: {checkpoint_path}"
            )
        resume_checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
    elif args.initialize_from:
        initialization_path = Path(args.initialize_from)
        if not initialization_path.is_file():
            raise FileNotFoundError(
                f"initialization checkpoint does not exist: {initialization_path}"
            )
        initialization_checkpoint = torch.load(
            initialization_path, map_location="cpu", weights_only=False
        )
        initialization = {
            "path": str(initialization_path.resolve()),
            "sha256": _sha256_file(initialization_path),
            "source_update": int(initialization_checkpoint.get("update", 0)),
            "source_environment_steps": int(
                initialization_checkpoint.get("environment_steps", 0)
            ),
            "source_scenario": initialization_checkpoint.get("scenario"),
        }

    source_checkpoint = resume_checkpoint or initialization_checkpoint
    if source_checkpoint is not None:
        saved_geometry = bool(
            source_checkpoint.get("args", {}).get("analytic_geometry", False)
        )
        if saved_geometry != args.analytic_geometry:
            raise ValueError(
                "--analytic-geometry must match the checkpoint architecture"
            )

    model = EntityActorCritic(
        analytic_geometry=args.analytic_geometry,
        kinematic_spec=TH05_KINEMATICS if args.analytic_geometry else None,
    ).cpu()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-5)
    if resume_checkpoint is not None:
        first_update, environment_steps = _apply_checkpoint(
            model, optimizer, resume_checkpoint, resume=True
        )
    elif initialization_checkpoint is not None:
        _apply_checkpoint(model, optimizer, initialization_checkpoint, resume=False)

    pool = WorkerPool(
        args.image,
        args.workers,
        args.frame_interval,
        args.seed,
        args.worker_timeout,
    )
    observation = pool.observations
    action_mask = pool.action_masks
    scenarios = [describe_th05_scenario(item) for item in observation]
    scenario = scenarios[0]
    hidden = torch.zeros(1, args.workers, model.hidden_size)
    objective_weights = np.asarray(args.objective_weights, dtype=np.float32)
    started = time.perf_counter()
    try:
        if any(item != scenario for item in scenarios[1:]):
            raise RuntimeError("rollout workers started with different TH05 scenarios")
        for update in range(first_update, first_update + args.updates):
            rollout_started = time.perf_counter()
            observations = np.empty(
                (args.rollout_steps, args.workers, FEATURE_DIM), dtype=np.float32
            )
            actions = np.empty((args.rollout_steps, args.workers), dtype=np.int64)
            log_probabilities = np.empty(
                (args.rollout_steps, args.workers), dtype=np.float32
            )
            values = np.empty(
                (args.rollout_steps, args.workers, 3), dtype=np.float32
            )
            rewards = np.empty_like(values)
            dones = np.empty((args.rollout_steps, args.workers), dtype=np.float32)
            hidden_states = np.empty(
                (args.rollout_steps, args.workers, model.hidden_size), dtype=np.float32
            )
            action_masks = np.empty(
                (args.rollout_steps, args.workers, 19), dtype=np.bool_
            )
            removed_probability_mass = 0.0
            successes = failures = no_miss_successes = 0

            model.eval()
            for step in range(args.rollout_steps):
                observations[step] = observation
                hidden_states[step] = hidden[0].numpy()
                action_masks[step] = action_mask
                with torch.no_grad():
                    logits, value, next_hidden = model.forward_step(
                        torch.from_numpy(observation), hidden
                    )
                    distribution = MaskedCategorical(
                        logits=logits,
                        valid_mask=(
                            torch.from_numpy(action_mask)
                            if args.hard_safety
                            else None
                        ),
                    )
                    action = distribution.sample()
                    log_probability = distribution.log_prob(action)
                    removed_probability_mass += float(
                        distribution.removed_probability_mass.sum().item()
                    )
                actions[step] = action.numpy()
                log_probabilities[step] = log_probability.numpy()
                values[step] = value.numpy()
                (
                    observation,
                    reward,
                    done,
                    flags,
                    action_mask,
                    step_no_miss_successes,
                ) = pool.step(actions[step])
                rewards[step] = reward
                dones[step] = done
                successes += int(np.count_nonzero(flags == 2))
                no_miss_successes += int(np.count_nonzero(step_no_miss_successes))
                failures += int(np.count_nonzero(flags == 1))
                hidden = next_hidden * torch.from_numpy(1.0 - done).view(1, -1, 1)

            with torch.no_grad():
                _, last_values_tensor, _ = model.forward_step(
                    torch.from_numpy(observation), hidden
                )
            last_values = last_values_tensor.numpy()
            rollout_seconds = time.perf_counter() - rollout_started
            environment_steps += args.rollout_steps * args.workers

            advantages, returns = _vector_gae(
                rewards,
                values,
                dones,
                last_values,
                args.gamma,
                args.gae_lambda,
            )
            scalar_advantages = advantages @ objective_weights
            scalar_advantages = (
                scalar_advantages - scalar_advantages.mean()
            ) / (scalar_advantages.std() + 1e-8)

            sequence_length = args.sequence_length
            feature_sequences = _as_sequences(observations, sequence_length)
            action_sequences = _as_sequences(actions, sequence_length)
            old_log_probability_sequences = _as_sequences(
                log_probabilities, sequence_length
            )
            advantage_sequences = _as_sequences(scalar_advantages, sequence_length)
            return_sequences = _as_sequences(returns, sequence_length)
            old_value_sequences = _as_sequences(values, sequence_length)
            done_sequences = _as_sequences(dones, sequence_length)
            action_mask_sequences = _as_sequences(action_masks, sequence_length)
            initial_hidden = np.swapaxes(hidden_states, 0, 1)[
                :, ::sequence_length
            ].reshape(-1, model.hidden_size)

            sequence_count = len(feature_sequences)
            update_started = time.perf_counter()
            totals = {"policy": 0.0, "value": 0.0, "entropy": 0.0, "kl": 0.0}
            batches = 0
            model.train()
            stop_early = False
            for _ in range(args.epochs):
                permutation = torch.randperm(sequence_count)
                epoch_kls = []
                for offset in range(0, sequence_count, args.sequence_batch_size):
                    indices = permutation[offset : offset + args.sequence_batch_size].numpy()
                    features_batch = torch.from_numpy(feature_sequences[indices])
                    actions_batch = torch.from_numpy(action_sequences[indices]).reshape(-1)
                    old_log_batch = torch.from_numpy(
                        old_log_probability_sequences[indices]
                    ).reshape(-1)
                    advantage_batch = torch.from_numpy(
                        advantage_sequences[indices]
                    ).reshape(-1)
                    return_batch = torch.from_numpy(return_sequences[indices]).reshape(-1, 3)
                    old_value_batch = torch.from_numpy(
                        old_value_sequences[indices]
                    ).reshape(-1, 3)
                    done_batch = torch.from_numpy(done_sequences[indices])
                    action_mask_batch = torch.from_numpy(
                        action_mask_sequences[indices]
                    ).reshape(-1, 19)
                    hidden_batch = torch.from_numpy(initial_hidden[indices]).unsqueeze(0)

                    logits, new_values = model.forward_sequence(
                        features_batch, hidden_batch, done_batch
                    )
                    distribution = MaskedCategorical(
                        logits=logits,
                        valid_mask=action_mask_batch if args.hard_safety else None,
                    )
                    new_log = distribution.log_prob(actions_batch)
                    log_ratio = new_log - old_log_batch
                    ratio = log_ratio.exp()
                    policy_loss = -torch.minimum(
                        ratio * advantage_batch,
                        torch.clamp(
                            ratio, 1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon
                        )
                        * advantage_batch,
                    ).mean()

                    clipped_values = old_value_batch + torch.clamp(
                        new_values - old_value_batch,
                        -args.value_clip,
                        args.value_clip,
                    )
                    value_loss = 0.5 * torch.maximum(
                        torch.square(new_values - return_batch),
                        torch.square(clipped_values - return_batch),
                    ).mean()
                    entropy = distribution.entropy().mean()
                    loss = (
                        policy_loss
                        + args.value_coefficient * value_loss
                        - args.entropy_coefficient * entropy
                    )

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()

                    approximate_kl = float(((ratio - 1.0) - log_ratio).mean().item())
                    epoch_kls.append(approximate_kl)
                    totals["policy"] += float(policy_loss.item())
                    totals["value"] += float(value_loss.item())
                    totals["entropy"] += float(entropy.item())
                    totals["kl"] += approximate_kl
                    batches += 1
                if epoch_kls and np.mean(epoch_kls) > args.target_kl:
                    stop_early = True
                    break

            update_seconds = time.perf_counter() - update_started
            metrics = {
                "update": update + 1,
                "environment_steps": environment_steps,
                "rollout_s": round(rollout_seconds, 4),
                "update_s": round(update_seconds, 4),
                "steps_per_s": round(
                    args.rollout_steps * args.workers / rollout_seconds, 2
                ),
                "policy_loss": round(totals["policy"] / batches, 6),
                "value_loss": round(totals["value"] / batches, 6),
                "entropy": round(totals["entropy"] / batches, 6),
                "approx_kl": round(totals["kl"] / batches, 6),
                "reward_mean": np.mean(rewards, axis=(0, 1)).round(6).tolist(),
                "death_events": int(np.count_nonzero(rewards[..., 0] < -0.1)),
                "action_frequency": (
                    np.bincount(actions.reshape(-1), minlength=19) / actions.size
                ).round(4).tolist(),
                "successes": successes,
                "no_miss_successes": no_miss_successes,
                "failures": failures,
                "early_stop": stop_early,
                "analytic_geometry": args.analytic_geometry,
                "hard_safety": args.hard_safety,
                "scenario": scenario,
                "initialization": initialization,
                "constrained_step_fraction": round(
                    float(np.mean(np.any(~action_masks, axis=-1)))
                    if args.hard_safety
                    else 0.0,
                    6,
                ),
                "removed_probability_mass": round(
                    removed_probability_mass / (args.rollout_steps * args.workers)
                    if args.hard_safety
                    else 0.0,
                    6,
                ),
                "wall_s": round(time.perf_counter() - started, 3),
            }
            print(json.dumps(metrics, sort_keys=True), flush=True)
            with metrics_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(metrics, sort_keys=True) + "\n")
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "update": update + 1,
                "environment_steps": environment_steps,
                "args": vars(args),
                "scenario": scenario,
                "initialization": initialization,
            }
            torch.save(checkpoint, checkpoint_path)
            if args.snapshot_every and (update + 1) % args.snapshot_every == 0:
                snapshot_dir = Path(args.snapshot_dir)
                snapshot_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    checkpoint,
                    snapshot_dir / f"update_{update + 1:06d}.pt",
                )
    finally:
        pool.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--sequence-batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--frame-interval", type=float, default=0.036)
    parser.add_argument("--worker-timeout", type=float, default=30.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--value-clip", type=float, default=0.2)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--entropy-coefficient", type=float, default=0.02)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument(
        "--analytic-geometry",
        action="store_true",
        help="append constant-velocity collision geometry inside the set encoder",
    )
    parser.add_argument(
        "--hard-safety",
        action="store_true",
        help="renormalize the policy over adapter-certified valid actions",
    )
    parser.add_argument(
        "--objective-weights",
        type=float,
        nargs=3,
        default=DEFAULT_REWARD_WEIGHTS.tolist(),
    )
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--checkpoint", default="models/pc98_entity_ppo.pt")
    parser.add_argument("--metrics", default="runs/pc98rl/metrics.jsonl")
    parser.add_argument("--snapshot-dir", default="models/pc98rl_snapshots")
    parser.add_argument("--snapshot-every", type=int, default=1)
    continuation = parser.add_mutually_exclusive_group()
    continuation.add_argument(
        "--resume",
        action="store_true",
        help="continue the same run, including optimizer and counters",
    )
    continuation.add_argument(
        "--initialize-from",
        help="start a new run from policy weights only; reset optimizer and counters",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
