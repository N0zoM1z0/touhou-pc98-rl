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
- A 180,136-parameter entity-set encoder, GRU policy, and three-value critic.
- Recurrent PPO with vector GAE, value clipping, KL stopping, snapshots, and
  multi-seed lower-confidence-bound checkpoint selection.
- Eight-emulator CPU rollout at about 223 transitions/s on this host.

TH01-04 are explicitly not claimed yet. They need their own verified memory and
termination adapters behind the same environment interface.

## Quick start

After building DOSBox-X and extracting your legally obtained game files:

```bash
git clone https://github.com/touhourl/th05patch.git external/th05patch
scripts/patch_th05.sh external/source/KAIKI external/th05patch external/th05-patched
scripts/prepare_hdi.sh \
  "external/game/[th01-05] 旧五作 (模拟器+汉化版+日文版)/日文版/zun.hdi" \
  external/th05-patched external/th05-rl.hdi

uv sync --reinstall-package touhou-pc98-rl
make test

export PATH="$PWD/external/dosbox-x-install/bin:$PATH"
CUDA_VISIBLE_DEVICES='' nice -n 10 taskset -c 0-47 \
  xvfb-run --auto-servernum uv run python -m pc98rl.ppo \
  --image external/th05-rl.hdi --workers 8 --threads 8
```

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
