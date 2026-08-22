"""Small, dependency-light latency reporting helpers."""

from __future__ import annotations

import math

import numpy as np


def latency_summary(milliseconds: list[float]) -> dict[str, float | int]:
    """Summarize positive wall-clock samples without discarding tail latency."""
    samples = np.asarray(milliseconds, dtype=np.float64)
    if samples.ndim != 1 or len(samples) == 0:
        raise ValueError("latency samples must be a non-empty one-dimensional list")
    if not np.isfinite(samples).all() or np.any(samples < 0.0):
        raise ValueError("latency samples must be finite and non-negative")
    return {
        "samples": int(len(samples)),
        "mean_ms": round(float(samples.mean()), 6),
        "p50_ms": round(float(np.percentile(samples, 50)), 6),
        "p95_ms": round(float(np.percentile(samples, 95)), 6),
        "p99_ms": round(float(np.percentile(samples, 99)), 6),
        "max_ms": round(float(samples.max()), 6),
    }


def deadline_utilization(latency_ms: float, deadline_ms: float) -> float:
    """Return the fraction of a real-time decision budget consumed."""
    if not math.isfinite(latency_ms) or latency_ms < 0.0:
        raise ValueError("latency_ms must be finite and non-negative")
    if not math.isfinite(deadline_ms) or deadline_ms <= 0.0:
        raise ValueError("deadline_ms must be finite and positive")
    return latency_ms / deadline_ms


__all__ = ["deadline_utilization", "latency_summary"]
