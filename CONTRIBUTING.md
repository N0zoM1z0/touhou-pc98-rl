# Contributing

Keep changes CPU-reproducible and separate game-specific discovery from the
shared environment and learning interfaces. A useful report includes the exact
game executable hashes, DOSBox-X version, Python/PyTorch versions, CPU affinity,
seed set, training transitions, and both validation and held-out evaluation.

Do not commit game images, patched executables, checkpoints, or personal data.
New game adapters need parser tests plus a live smoke-test recipe. Algorithm
changes should report paired baselines rather than a single favorable run.

Contributions are distributed under GPL-3.0-or-later; preserve attribution for
code derived from other projects.
