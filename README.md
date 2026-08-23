# Touhou PC-98 RL

A CPU-first reinforcement-learning laboratory for Touhou 1-5. The current
working game adapter is TH05: DOSBox-X boots a private patched HDI for every
environment, Rust reads structured game state directly from the child process,
and a compact recurrent PPO agent learns without a GPU.

This is a new framework, not a renamed entry point for the reference trainer.
The Python package is `pc98rl`, its native extension is `pc98rl._native`, and the
training/evaluation path is designed around reproducible CPU experiments.

## What works now

- Reproducible patching and HDI assembly for the supplied Japanese TH05 image.
- Exact child-process attachment with current DOSBox-X PC-98 emulation.
- A Gymnasium environment with independent 21 MiB image copies per worker.
- A 273-float compact observation path that skips dense spatial-map allocation.
- A roughly 180k-parameter entity-set encoder, GRU policy, and three-value
  critic.
- Recurrent PPO with vector GAE, value clipping, KL stopping, snapshots, and
  multi-seed lower-confidence-bound checkpoint selection.
- Optional analytic relative-motion features and adapter-certified hard action
  masks applied consistently during rollout and optimization.
- Resident-counter miss events, an explicit miss-only training cost, and an
  experimental policy-accounted emergency-bomb mask.
- A transactional 36-ms action path, an optional audited deathbomb guard for
  historical ablations, and a strict NMNB mode that permanently masks bombs.
- Offline outcome and future-safety teachers that update only the actor head,
  plus an inference-only exporter that removes optimizer and auxiliary-head
  state from deployment artifacts.
- An offline-only DOSBox-X save-state brancher with exact guest-frame stepping;
  repeated action sequences reproduce compact and raw game state byte-for-byte.
- Grouped multi-anchor exact counterfactual datasets, trajectory-level
  train/selection/held-out boundaries, and actor-head-only causal distillation.
- Eight-emulator CPU rollout at about 223 transitions/s on this host.

TH01-04 are explicitly not claimed yet. The immediate target is a TH05 Lunatic
no-miss, no-bomb (NMNB) clear; cross-game transfer remains a later research
question requiring verified per-game adapters.

## Quick start

After building DOSBox-X and extracting your legally obtained game files:

```bash
git clone https://github.com/touhourl/th05patch.git external/th05patch
scripts/patch_th05.sh external/source/KAIKI external/th05patch external/th05-patched
scripts/prepare_hdi.sh \
  "external/game/[th01-05] 旧五作 (模拟器+汉化版+日文版)/日文版/zun.hdi" \
  external/th05-patched external/th05-rl.hdi
scripts/make_th05_scenario.sh \
  external/th05-rl.hdi external/th05-lunatic-stage1.hdi \
  --stage 1 --rank 3 --character 2 --power 0 --lives 3 --bombs 3

uv sync --reinstall-package touhou-pc98-rl
make test

CUDA_VISIBLE_DEVICES='' nice -n 10 taskset -c 0-47 \
  xvfb-run --auto-servernum uv run python -m pc98rl.ppo \
  --image external/th05-lunatic-stage1.hdi --workers 8 --threads 8 \
  --analytic-geometry --hard-safety --no-allow-bombs
```

The environment discovers the repository-local DOSBox-X build automatically.
Set `PC98RL_DOSBOX_X=/absolute/path/to/dosbox-x` to select another build.

Exporting an offline-trained checkpoint makes the deployment boundary explicit:

```bash
uv run python scripts/export_online_actor.py \
  --checkpoint models/distill/training-checkpoint.pt \
  --output models/deploy/online-actor.pt \
  --regular-bullet-safety-horizon 6 \
  --no-regular-bullet-least-risk-fallback \
  --no-allow-bombs --no-deathbomb-safety
```

The exported file retains the small recurrent policy and audit metadata, but
contains no optimizer, training-only risk head, or trajectory path list.
The evaluator reports `bomb_actions` and `nmnb_success`; NMNB checkpoints reject
automatic bomb mechanisms instead of silently relying on them.

See [CPU_RL.md](CPU_RL.md) for the full setup, benchmark methodology, honest
short-run results, evaluation commands, known failure modes, and next research
experiments. The evolving English research manuscript is in
[paper/main.tex](paper/main.tex).

## Design boundary

Game images, patched executables, models, and run logs are intentionally ignored
by Git. The repository contains only code, configuration, tests, and measured
experiment notes. Training defaults force the PyTorch CPU wheel; no CUDA or XPU
path is required.

## Provenance and license

The low-level TH05 reader began from published GPL-3.0-or-later work in
[`touhourl/thrl`](https://github.com/touhourl/thrl). This repository replaces its
training stack and project identity while retaining the applicable license and
attribution. See [NOTICE.md](NOTICE.md). Game data is not included.
