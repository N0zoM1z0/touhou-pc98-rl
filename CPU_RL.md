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
| Live control | One timing-racy memory write; learner time advances the game | 36 ms transactions, 1 ms action refresh, paused learner boundary, native deathbomb guard |

The reference column documents the baseline used during design; it is not this
project's runtime or public interface. The compact reader does not build and discard dense maps. Its live paused-game
median was 0.516 ms versus 3.550 ms for the dense reader: 6.88x faster and 776x
less float transfer. The entity model is 13.2x smaller. On this host, eight
games produced about 223 transitions/s; collecting 2,048 transitions took about
9.19 s and a three-epoch PPO update took about 0.20-0.33 s.

The 223 transitions/s figure predates transactional pause/action refresh and is
retained as a reader/model baseline; the current real-time collector must be
rebenchmarked separately. These comparisons establish that the new execution and learning infrastructure
is substantially more CPU-efficient. They do **not yet establish stronger final
gameplay**: the current training budget is small and no compatible reference
checkpoint is available for a controlled head-to-head evaluation.

## Reproduce the local game image

Required system tools are `bspatch` (`bsdiff` package), `mtools`, a Rust
toolchain, `uv`, and a locally built recent DOSBox-X. The tested emulator is
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

CUDA_VISIBLE_DEVICES='' nice -n 10 taskset -c 0-7 \
  uv run python scripts/evaluate_policy.py \
  --image external/th05-rl.hdi --policy random --steps 200 --seed 41
```

The environment first uses `PC98RL_DOSBOX_X` when set, then searches `PATH`,
then checks `external/dosbox-x-install/bin/dosbox-x`. It passes the resolved
absolute executable to each native child spawn. In a displayless shell the
native launcher selects SDL's dummy video backend automatically; Xvfb is not a
runtime requirement. Early emulator exit is reported directly instead of being
misdiagnosed as a process-memory permission failure.

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
`--force` is explicit. `--stage` is the human-facing regular-stage number;
the script converts Stage 1--6 to the game's zero-based resident value 0--5.
Use `--stage extra` for resident value 6. This conversion is deliberate: the
patch README describes `skip_to=2` as Stage 2 even though its loader assigns the
value directly to the zero-based resident field. Rank 3 is Lunatic. Models and
reports record both the display stage and internal index rather than relying on
a filename.

## Train without disturbing other workloads

The trainer limits each emulator worker to one PyTorch thread and defaults the
learner to eight threads. More threads were not better here: model-update timing
improved through eight threads, while 16 threads caused severe oversubscription.
Use `taskset` to reserve only known-idle cores and a positive niceness value:

```bash
CUDA_VISIBLE_DEVICES='' nice -n 10 taskset -c 0-47 \
  xvfb-run --auto-servernum uv run python -m pc98rl.ppo \
  --image external/th05-rl.hdi \
  --workers 8 --threads 8 \
  --rollout-steps 256 --updates 100 \
  --analytic-geometry --hard-safety \
  --deathbomb-safety \
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
ablated against the same transition budget. Workers track deaths across rollout
boundaries and log both `successes` and `no_miss_successes` for completed
episodes.

Reset waits until TH05's initial invincibility/input lock ends. The older
readable-memory gate exposed roughly 80 uncontrollable transitions per reset
and assigned their survival reward to actions that the game could not execute.

The live game is paused while the policy or learner is working. During each
36 ms action transaction, the command is refreshed every 1 ms because TH05
rebuilds its input word from the physical keyboard every native frame. With
`--deathbomb-safety`, a tiny native guard checks the exact pending-hit/miss-timer
predicate on that refresh path and can preempt a policy action without waiting
for Python inference. This is the online control plane; PPO updates, replay
analysis, and future larger teachers belong to the paused learning plane.

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
  --baseline untrained --seeds 51 52 53 54 --steps 900 --jobs 4
```

Do not trust the last PPO update. Select snapshots on several fixed seeds.
Selection is lexicographic: clears, no-miss clears, then a lower-confidence-bound
return score:

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

### Matched Lunatic ablation

We trained plain compact recurrent PPO and analytic-geometry plus hard-safety
PPO with three matched training seeds. Each run used 24,576 transitions and
retained updates 4, 8, and 12. Nine candidates per condition were selected on
seeds 71--73, then compared on untouched seeds 81--84 for 1,200 steps. A source
audit after training established that this image was Stage 2 (resident index 1),
not Stage 1 as the patch README implied.

| Selected condition | Validation mean / LCB | Test mean return | Test deaths | Clears |
| --- | ---: | ---: | ---: | ---: |
| Plain PPO | -0.856 / -1.449 | -1.648 | 14 | 0 |
| Geometry + hard safety | -0.822 / -1.422 | -1.054 | 11 | 0 |

The paired test gain is `+0.594` with standard error `0.669`. One seed accounts
for most of the gain, so this is not a statistically persuasive gameplay win.
The hard mask constrained 40.1% of decision points while removing only 2.47% of
policy probability on average; it enforces exact availability and boundary
semantics, not competent dodging.

A separate true Stage 1 Lunatic probe started at boss phase 3 and terminated at
phase 4 with full power. A hard-safety random policy cleared it in 397 steps
with one death. This validates phase success and provides a dense starting task
for completion-gated curriculum training. Exact values are in
`experiments/2026-08-22-th05-lunatic-ablation.json`.

### First learned curriculum rung

Two geometry+safety PPO pilots trained the Stage 1 phase 3-to-4 task. Snapshot
selection over seeds 111--114 chose the phase-tuned update 8 with 4/4 clears,
3/4 no-miss clears, and 0.25 mean deaths. On untouched seeds 121--128, both the
selected and matched untrained policies cleared 8/8, but learning improved
no-miss clears from 1/8 to 4/8, reduced deaths from 7 to 4, and raised mean
return from 1.661 to 1.952. The paired gain was `+0.290 +/- 0.100` standard
error.

The entropy/learning-rate tuning itself did **not** replicate. On a second
held-out set (seeds 131--138), default PPO update 8 achieved 3/8 no-miss clears
versus 1/8 for the tuned update, with tuned-minus-default return
`-0.093 +/- 0.136`. The supported conclusion is that phase curriculum enables
learned safety; four validation seeds are insufficient for robust
hyperparameter selection.

Frequent phase terminals also reduced mean collection throughput to 76--80
steps/s from the 223 steps/s no-reset rate. This is direct evidence for the
planned asynchronous sequence-block collector. The full machine-readable record
is `experiments/2026-08-22-th05-lunatic-curriculum.json`.

### Second curriculum rung and sparse-miss limit

The frozen phase 3-to-4 checkpoint transferred to the unseen phase 2-to-4 task:
on seeds 211--218 it cleared 7/8 versus 6/8 for a matched untrained policy and
reduced deaths from 19 to 16, but neither policy achieved a no-miss clear. A new
weights-only curriculum initializer then loaded the policy while resetting Adam
moments, learning rate, counters, and recurrent state. Its provenance record
includes the source checkpoint SHA-256 and scenario.

Default PPO fine-tuning used seed 6066 for 32,768 transitions. Validation over
updates 4/8/12/16 on seeds 301--304 selected update 4 by the frozen
clear/no-miss/return-LCB rule. On untouched seeds 311--318, the selected policy
improved from 6/8 to 8/8 clears, reduced deaths from 19 to 14, and gained
`+0.870 +/- 0.464` paired return versus its initialization policy. Both remained
at 0/8 no-miss. The supported conclusion is improved phase completion, not
perfect play.

This result exposes a different bottleneck: a miss is sparse and only marks the
collision step even though the responsible movement may begin tens of actions
earlier. The next ablation will add training-only, multi-horizon future-miss
prediction to the shared representation. Labels come from the same on-policy
rollouts, are masked when the horizon is censored, and are never inputs at
inference. PPO and PPO+auxiliary must use matched transitions and report clears,
no-miss clears, and deaths; a completion regression falsifies the proposed
improvement. Exact selection and test reports are under `experiments/` with the
`th05-lunatic-p2` prefix.

### Auxiliary future-miss ablation: negative result

The proposed auxiliary head was implemented with action-conditioned 15/30-step
labels, episode-boundary truncation, rollout censoring, class-balanced BCE, and
preserved policy-sampling RNG. A matched-budget pilot initialized both
conditions from the selected phase 2-to-4 policy and trained each for 16,384
transitions with seed 8088. The auxiliary coefficient was fixed at `0.05` before
validation.

Training totals were misleading: auxiliary training observed 51 deaths and 4
failures versus 102 and 13 for plain PPO. Validation seeds 501--504 nevertheless
selected update 4 for both conditions and strongly favored plain PPO. On
untouched seeds 511--518, plain PPO cleared 8/8 with 14 deaths and mean return
2.455; auxiliary PPO cleared 7/8 with 18 deaths and mean return 2.065. Both had
zero no-miss clears. Auxiliary-minus-plain paired return was
`-0.390 +/- 0.312` standard error.

This rejects the current auxiliary mechanism. Vector GAE already propagates
death rewards backward, while an on-policy future label is not counterfactual:
it describes the outcome under the whole sampled future action sequence but
does not identify a safer current action. Shared-encoder gradients can therefore
distort completion features without adding actionable information. The next
survival experiment must test an action-contrastive risk target or an explicit
survival constraint over multiple training seeds, not merely tune this loss on
the test set. Exact results are in
`experiments/2026-08-22-th05-lunatic-auxiliary-ablation.json`.

### Explicit miss cost and deployment-policy result

The native reward originally combined terminal success, terminal failure,
per-step survival, and a miss in one survival component. Merely increasing its
objective weight therefore could not make one miss lexicographically worse than
one clear. The environment now detects misses from the resident miss counter,
rather than inferring them from a negative reward, and PPO can add an explicit
scaled cost with `--extra-miss-penalty` without changing the three-head model or
invalidating existing policy weights.

A phase 2-to-4 fine-tune initialized from the selected curriculum checkpoint,
used penalty `2.0`, seed 9191, and 32,768 transitions. Validation seeds 621--624
selected update 4: 4/4 clears, zero no-miss clears, 1.75 mean deaths, and return
LCB 2.405. Updates 8 and 16 already regressed to 2/4 and 3/4 clears.

On untouched stochastic seeds 631--638, the initializer achieved 5/8 clears,
18 misses, and mean return 1.403. The selected miss-cost policy achieved 7/8
clears, 16 misses, and mean return 2.099. Its paired return gain was
`+0.696 +/- 0.780` standard error. Both remained at 0/8 no-miss, so this is at
most a completion improvement and does not validate the intended perfect-play
mechanism.

Deterministic argmax deployment was decisively worse: both checkpoints failed
all eight seeds and exhausted all three lives in every run. The categorical
policy represents useful motion with several competing direction modes; taking
one mode at every step collapses that mixture into a few repeated actions.
Training exploration and deployment cannot therefore be separated with a raw
argmax rule for this architecture.

An optional last-resort bomb shield is also implemented as a policy-accounted
action mask. It only intervenes when resources remain, the player is vulnerable,
and constant-velocity geometry predicts a short-horizon close approach. A
four-seed calibration was inconclusive and produced no no-miss clear; 10 px / 6
frames had 4/4 clears and seven misses versus 3/4 and eight for no shield, while
18 px / 8 frames had 3/4 and nine. Because TH05's actual collision logic uses
square, projectile-specific hitboxes and bullet activation states, these
numbers justify auditing exact native collision semantics rather than widening
the approximate threshold. Full values are tracked in
`experiments/2026-08-22-th05-lunatic-explicit-miss-cost.json`.

### Audited regular-bullet safety: reliable clears, not perfect play

The native adapter now exposes an action mask derived from TH05's ReC98
collision rules. It uses the 8-by-8 regular-bullet killbox, the asymmetric
16/20-by-22/22 graze box, the requirement that a bullet be grazed in a previous
frame, cloud/decay activation states, per-character focused and unfocused
movement speeds, and discrete native-frame prediction. The policy remains
responsible for choosing among all certified movement actions and a legal bomb;
the shield does not script a route. Special projectiles remain outside this
mask and are an explicit limitation.

A corrected four-seed frozen-policy calibration on seeds 675--678 improved
completion from 2/4 to 4/4, reduced deaths from nine to eight, and increased
mean return from 1.223 to 2.442 with 259 interventions. This justified training
under the same mask rather than deploying an unfamiliar constraint only after
training. A 32,768-transition fine-tune with seed 9393 selected update 8 on
validation seeds 681--684: 4/4 clears, 1.5 mean deaths, mean return 2.727, but
zero no-miss clears. Update 16 regressed to 3/4.

On untouched seeds 691--698, both the miss-cost initializer and the selected
shield-aware policy cleared 8/8. Deaths changed from 15 to 14 and mean return
from 2.458 to 2.618; paired return gain was `+0.161 +/- 0.150` standard error.
Both remained at 0/8 no-miss. The mask is therefore retained as a useful
curriculum scaffold, but the perfect-play hypothesis is not supported. The
next audit must cover deathbomb timing and special projectile hitboxes rather
than extending this PPO run indefinitely. Full values are in
`experiments/2026-08-22-th05-lunatic-audited-bullet-safety.json`.

A timing diagnostic also rejected `frame_interval_s=0` as a throughput trick.
The agent loop then outruns the DOSBox native frame clock and samples many
actions against repeated game states; 1,600 API steps do not cover a full
phase-2 episode. Until the adapter waits on an observed native-frame advance,
formal training and evaluation retain the paced 36 ms step interval.

### Transactional control and deathbomb audit

A one-shot action-18 write did not change bomb stock: TH05's native keyboard
refresh overwrote it before `player_bomb` observed it. Repeating the command 142
times over 80 ms reliably changed stock from three to two. The environment now
submits actions transactionally and freezes each emulator while the learner is
inactive, so learning latency cannot silently advance the game.

The native deathbomb guard follows the audited eight-frame window (`miss_time`
40 through 33, plus the pending-hit byte). In a fixed-policy phase 2-to-4 trace,
it cancelled collisions at steps 194, 509, and 757, consuming exactly the three
available bombs without a miss. A fourth fatal path began at step 950 with zero
bombs and registered at step 953. This proves the low-latency safety path works;
it also proves that safety cannot replace a movement policy that reaches four
fatal states. The event record is
`experiments/2026-08-22-th05-lunatic-realtime-deathbomb.json`.

### Offline future safety and the deployment boundary

The asymmetric architecture is now implemented. A training-only risk head uses
24 frozen-policy pre-boss trajectories and predicts native collision-preemption
or miss events at 8-, 24-, and 64-step horizons. It evaluates all 19 actions
offline, forms a risk-adjusted target distribution, and distills only the actor
head. The deployment exporter removes that risk head, optimizer state, and
trajectory paths; the resulting 180,456-parameter online actor is about 742 kB.

On untouched pre-boss seeds, this future-safety actor improves no-miss clears
from 0/8 to 2/8 and reduces misses from nine to seven. On untouched complete
Stage 1 seeds, it improves ordinary completion from 4/8 to 7/8, but both systems
remain at 0/8 no-miss and the student's total misses increase from 12 to 13.
The actor is therefore useful as a curriculum-completion initializer, not yet a
perfect-survival solution. A latest-predicted-collision safety fallback and
longer H10/H16 masks also fail selection and are rejected.

The complete online path---native shields, recurrent actor, and categorical
sampling---has 1.474 ms p99 latency on one CPU thread, or 4.09% of the 36-ms
transaction interval. The emulator is paused during inference, so offline
teacher complexity cannot consume native game time. Full values and hashes are
in `experiments/2026-08-22-th05-lunatic-stage-boundary.json`.

## Next experiments

The highest-value next work is:

1. Branch from paused emulator save states near audited high-risk decisions,
   evaluate every legal action under matched short continuations, and distill
   the resulting action-contrastive collision-time labels into the unchanged
   deadline-bounded actor.
2. Add compact normal-enemy tokens as a separately ablated information change;
   do not confound it with the counterfactual-supervision experiment.
3. Audit special-projectile collision boxes before widening any predictive mask.
4. Synchronize each environment step to an observed native-frame advance, then
   implement a reset-aware sequence-block collector and verify its on-policy
   age bound before using its throughput result in algorithm comparisons.
5. Continue Stage 1 from its true low-power start, requiring held-out no-miss
   completion rather than ordinary completion before advancing to later stages.
6. Implement TH01-04 adapters behind the same Gymnasium interface and add
   per-game contract tests before claiming full old-five-games support.

The official [ReC98 P0335 release](https://github.com/nmlgc/ReC98/releases/tag/P0335)
was used as a clean-game boot baseline. Building ReC98 from source still depends
on the historical proprietary Borland/TASM toolchain, so this project uses its
published binaries only for validation and does not make them a build dependency.
