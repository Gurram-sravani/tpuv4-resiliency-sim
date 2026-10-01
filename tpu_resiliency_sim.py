#!/usr/bin/env python3
"""Small Monte Carlo simulator for static and reconfigurable TPU allocation.

The model is intentionally abstract: a cube is the unit of allocation. A
static system needs one fixed block of usable cubes; a reconfigurable system
can connect any usable cubes in the pod.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt


def make_states(n, failure_rate, occupied_rate, pattern, rng):
    """Return a list whose values are free, failed, or occupied."""
    states = ["free"] * n
    failed = {i for i in range(n) if rng.random() < failure_rate}
    for i in failed:
        states[i] = "failed"

    candidates = [i for i in range(n) if i not in failed]
    target = min(len(candidates), round(occupied_rate * n))
    if pattern == "random":
        chosen = rng.sample(candidates, target)
    elif pattern == "concentrated":
        chosen = candidates[:target]
    elif pattern == "fragmented":
        # Spread occupied cubes across the pod before filling a second round.
        chosen = []
        stride = max(1, len(candidates) // max(1, target))
        cursor = 0
        while len(chosen) < target:
            idx = candidates[cursor % len(candidates)]
            if idx not in chosen:
                chosen.append(idx)
            cursor += stride
            if cursor > len(candidates) * 2:
                for idx in candidates:
                    if idx not in chosen:
                        chosen.append(idx)
                    if len(chosen) == target:
                        break
    else:
        raise ValueError(f"Unknown occupancy pattern: {pattern}")

    for i in chosen:
        states[i] = "occupied"
    return states


def static_success(states, job_size):
    """Static system: one fixed contiguous block must be entirely usable."""
    for start in range(0, len(states), job_size):
        block = states[start : start + job_size]
        if len(block) == job_size and all(s == "free" for s in block):
            return True
    return False


def reconfigurable_success(states, job_size):
    """Reconfigurable system: any healthy and free cubes may be connected."""
    return sum(s == "free" for s in states) >= job_size


def estimate(n, job_size, failure_rate, occupied_rate, pattern, trials, seed):
    rng = random.Random(seed)
    static = []
    flexible = []
    for _ in range(trials):
        states = make_states(n, failure_rate, occupied_rate, pattern, rng)
        static.append(static_success(states, job_size))
        flexible.append(reconfigurable_success(states, job_size))
    return {
        "static_success": mean(static),
        "reconfigurable_success": mean(flexible),
        "n_cubes": n,
        "job_size": job_size,
        "failure_rate": failure_rate,
        "occupied_rate": occupied_rate,
        "pattern": pattern,
        "trials": trials,
    }


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_lines(path, rows, x_key, xlabel, title):
    path.parent.mkdir(parents=True, exist_ok=True)
    grouped = {}
    for row in rows:
        grouped.setdefault(row["series"], []).append(row)
    plt.figure(figsize=(7.2, 4.5))
    for label, values in grouped.items():
        values.sort(key=lambda r: r[x_key])
        plt.plot([r[x_key] for r in values],
                 [100 * r["success"] for r in values],
                 marker="o", linewidth=2, label=label)
    plt.xlabel(xlabel)
    plt.ylabel("Successful job requests (%)")
    plt.title(title)
    plt.ylim(0, 105)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def run_all(outdir, trials, seed):
    outdir.mkdir(parents=True, exist_ok=True)
    n = 64
    rows1 = []
    for k in [1, 2, 4, 8, 16, 32, 48, 64]:
        result = estimate(n, k, 0.01, 0.0, "random", trials, seed + k)
        rows1 += [
