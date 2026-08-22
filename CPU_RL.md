# CPU-only Touhou PC-98 RL

## Current status

The end-to-end TH05 path is working on CPU: a patched HDI boots in DOSBox-X,
the Rust extension finds the live resident state in current DOSBox-X, actions
reach the game, the Gymnasium environment resets private game copies, and the
recurrent PPO trainer runs eight emulators in parallel.

This is currently a **TH05 implementation and a TH01-05 framework foundation**.
The reference repository only has a complete modern reader/patch path for TH05;
TH01-04 still need game-specific memory schemas, patch/termination adapters, and
validation. No copyrighted game image or patched executable is tracked by Git.

## Architecture

| Area | Reference path | CPU path in this branch |
| --- | --- | --- |
| Observation | 273 floats plus `24 x 92 x 96` dense maps | 273 floats with two masked nearest-entity sets |
| Per-step map traffic | 211,968 floats, about 0.81 MiB | No dense map; about 1.07 KiB of features |
| Model | CNN + GRU, 2,375,478 parameters | Set encoder + GRU, 180,136 parameters |
| Runtime | Accelerator-oriented trainer | Explicit CPU wheels and bounded PyTorch threads |
| Environment | Custom worker/trainer coupling | Gymnasium `Env` with isolated HDI per instance |
| Process discovery | First matching DOSBox process | Exact spawned PID, optional explicit PID attach |
| Model retention | Latest checkpoint | Periodic snapshots plus multi-seed LCB selector |

The reference column documents the baseline used during design; it is not this
project's runtime or public interface. The compact reader does not build and discard dense maps. Its live paused-game
median was 0.516 ms versus 3.550 ms for the dense reader: 6.88x faster and 776x
less float transfer. The entity model is 13.2x smaller. On this host, eight
games produced about 223 transitions/s; collecting 2,048 transitions took about
9.19 s and a three-epoch PPO update took about 0.20-0.33 s.

These comparisons establish that the new execution and learning infrastructure
is substantially more CPU-efficient. They do **not yet establish stronger final
gameplay**: the current training budget is small and no compatible reference
checkpoint is available for a controlled head-to-head evaluation.

## Reproduce the local game image

Required system tools are `bspatch` (`bsdiff` package), `mtools`, `xvfb-run`, a
Rust toolchain, `uv`, and a locally built recent DOSBox-X. The tested emulator is
official DOSBox-X tag `dosbox-x-v2026.07.02`; its PC-98 mode is documented in the
[official DOSBox-X PC-98 guide](https://github.com/joncampbell123/dosbox-x/wiki/Guide%3APC%E2%80%9098-emulation-in-DOSBox%E2%80%90X).

Clone the executable patch repository, then patch the extracted Japanese TH05
directory from the supplied old-games archive:

```bash
git clone https://github.com/touhourl/th05patch.git external/th05patch
scripts/patch_th05.sh \
  external/source/KAIKI \
  external/th05patch \
  external/th05-patched

scripts/prepare_hdi.sh \
  "external/game/[th01-05] 旧五作 (模拟器+汉化版+日文版)/日文版/zun.hdi" \
  external/th05-patched \
  external/th05-rl.hdi
```

The local archive contains an unusual mixed executable revision. Its hashes are:

```text
c41f6e6b...  MAIN.EXE  (works with th05patch main layout 1)
c94efc07...  OP.EXE    (manifest layout 2)
5044ae03...  ZUN.COM   (manifest layout 1)
```

`scripts/patch_th05.sh` verifies the complete hashes and intentionally applies
that mixed layout. Applying matching-number variants as a pair is wrong for this
archive: main layout 2 boots to a black screen. The script fails closed for any
unknown source revision.

The default curriculum file starts TH05 stage 1 with three lives, three bombs,
zero power, character 2, and rank 0. Edit `config/th05_cpu/KAIKII.CFG` before
preparing the HDI to change that curriculum.

## Build and smoke test

```bash
uv sync
cargo test --lib
uv sync --reinstall-package touhou-pc98-rl

export PATH="$PWD/external/dosbox-x-install/bin:$PATH"
CUDA_VISIBLE_DEVICES='' nice -n 10 taskset -c 0-7 \
  xvfb-run --auto-servernum uv run python scripts/evaluate_policy.py \
  --image external/th05-rl.hdi --policy random --steps 200 --seed 41
```

`ptrace_scope` must allow reading the child DOSBox process. Check it with:

```bash
cat /proc/sys/kernel/yama/ptrace_scope
```

The tested host reports `0`. Changing a stricter system policy is an
administrator decision and is not performed by these scripts.

## Build an explicit TH05 scenario

Never mutate the verified template image for a curriculum experiment. Create a
named copy whose embedded configuration is range-checked:

```bash
scripts/make_th05_scenario.sh \
  external/th05-rl.hdi external/th05-lunatic-stage1.hdi \
  --stage 1 --phase 0 --end-phase 0 \
  --character 2 --rank 3 --power 0 --lives 3 --bombs 3
```

The script refuses to overwrite its input or an existing output unless
`--force` is explicit. Stage uses the patch's `skip_to` numbering (1 is the
first regular stage), and rank 3 is Lunatic. Models and reports must record the
complete scenario rather than relying on a filename.

## Train without disturbing other workloads

The trainer limits each emulator worker to one PyTorch thread and defaults the
learner to eight threads. More threads were not better here: model-update timing
improved through eight threads, while 16 threads caused severe oversubscription.
Use `taskset` to reserve only known-idle cores and a positive niceness value:

```bash
export PATH="$PWD/external/dosbox-x-install/bin:$PATH"
CUDA_VISIBLE_DEVICES='' nice -n 10 taskset -c 0-47 \
  xvfb-run --auto-servernum uv run python -m pc98rl.ppo \
  --image external/th05-rl.hdi \
  --workers 8 --threads 8 \
  --rollout-steps 256 --updates 100 \
  --analytic-geometry --hard-safety \
  --snapshot-every 2
```

Eight workers mean 2,048 game transitions per update. Each reset copies the
21 MiB template HDI to a private temporary directory, preventing workers from
racing on score and configuration writes. Run parallel top-level jobs under
different X displays; one `xvfb-run --auto-servernum` wrapper is sufficient for
the eight workers inside a single trainer.

`--analytic-geometry` derives bounded relative velocity, closing speed,
time-to-closest-approach, and miss-distance features inside the model. Physical
normalization is supplied by the TH05 adapter rather than embedded in PPO.
`--hard-safety` renormalizes the policy over adapter-certified actions. The
current mask only removes a bomb with no bomb stock and boundary-directed moves
that are equivalent under TH05's position clamp; it does not contain a scripted
collision-avoidance policy. Both switches are experimental and should be
ablated against the same transition budget.

Reset waits until TH05's initial invincibility/input lock ends. The older
readable-memory gate exposed roughly 80 uncontrollable transitions per reset
and assigned their survival reward to actions that the game could not execute.

Evaluate one checkpoint:

```bash
CUDA_VISIBLE_DEVICES='' nice -n 10 taskset -c 0-7 \
  xvfb-run --auto-servernum uv run python scripts/evaluate_policy.py \
  --image external/th05-rl.hdi --policy checkpoint \
  --checkpoint models/pc98_entity_ppo.pt --steps 1200 --seed 41
```

Run and record a paired multi-seed comparison:

```bash
CUDA_VISIBLE_DEVICES='' nice -n 10 taskset -c 0-7 \
  xvfb-run --auto-servernum uv run python scripts/compare_policies.py \
  --image external/th05-rl.hdi \
  --checkpoint models/pc98_entity_ppo_best.pt \
  --baseline untrained --seeds 51 52 53 54 --steps 900
```

Do not trust the last PPO update. Select snapshots on several fixed seeds with a
lower-confidence-bound score:

```bash
CUDA_VISIBLE_DEVICES='' nice -n 10 taskset -c 0-7 \
  xvfb-run --auto-servernum uv run python scripts/select_checkpoint.py \
  --image external/th05-rl.hdi --seeds 41 42 43 44 --steps 1200 --jobs 4 \
  models/pc98rl_snapshots/*.pt
```

This writes the selection report under ignored `runs/` and copies the selected
snapshot to ignored `models/pc98_entity_ppo_best.pt`. Each parallel evaluation
owns a private HDI copy and emulator process, so `--jobs` changes throughput
without coupling episode state. Keep it below the number of CPUs left idle by
other experiments.

## Short-run evidence (2026-08-22)

The delivered local best artifact is `models/pc98_entity_ppo_best.pt`, SHA-256
`b17810efdf0c87cd78771641537e7d26b7356609934f69e274f0630888530587`.
It is update 8 of the training-seed-202 run (16,384 transitions).

Six candidate snapshots were compared for 900 steps on selection seeds 41, 43,
and 44. The chosen checkpoint had mean return `+0.2743`, standard error
`0.2362`, LCB score `+0.0381`, and two deaths across the three runs.

The final paired stochastic test used untouched seeds 51-54:

| Policy | Mean scalar return | Total deaths | Mean resource objective |
| --- | ---: | ---: | ---: |
| Untrained network | +0.0529 | 4 | -140.75 |
| Selected update 8 | +0.3639 | 2 | -116.25 |

That is an absolute return gain of `+0.3110`, 50% fewer deaths, and a 17.4%
smaller resource cost on the untouched set. Boss-damage reward was essentially
flat and no run completed the stage within 900 steps, so this demonstrates
better short-prefix survival/resource management, not convergence or a stage clear.
The exact machine-readable record is
`experiments/2026-08-22-th05-cpu.json`.

An earlier exploratory 24,576-transition run (whose checkpoint was overwritten
before snapshot retention was added) reached mean return `+0.1096`, reduced
deaths from 7 to 6, and improved mean resource cost from `-207.25` to `-140.25`
on the same four seeds. Continuing that run to 49,152 transitions regressed to
about `-0.895` mean return. Three additional 16,384-transition training seeds
also varied widely. That result motivated snapshot retention and multi-seed
selection; it should not be presented as a reproducible delivered model.

The hand-written `SafetyHeuristic` was a negative ablation: it reached four
deaths and terminal failure around 948-1,107 steps in short tests, worse than the
random baseline. It is not used for behavior cloning or PPO warm-up.

## Next experiments

The highest-value next work is:

1. Run longer, multi-seed training with a separate validation seed set and keep
   a final untouched test set.
2. Add enemy tokens and time-to-collision features; the current 273-float schema
   includes nearest bullets/special projectiles but not a compact enemy set.
3. Randomize stage, phase, character, rank, and power through curriculum levels,
   then measure stage completion rather than only short-prefix return.
4. Decouple reset latency from synchronous rollout collection, or use an
   asynchronous worker pool once terminal events become frequent.
5. Implement TH01-04 adapters behind the same Gymnasium interface and add
   per-game contract tests before claiming full old-five-games support.

The official [ReC98 P0335 release](https://github.com/nmlgc/ReC98/releases/tag/P0335)
was used as a clean-game boot baseline. Building ReC98 from source still depends
on the historical proprietary Borland/TASM toolchain, so this project uses its
published binaries only for validation and does not make them a build dependency.
