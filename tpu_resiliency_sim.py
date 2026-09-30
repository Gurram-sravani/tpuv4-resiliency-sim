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
            {"job_size": k, "series": "Static", "success": result["static_success"]},
            {"job_size": k, "series": "Reconfigurable", "success": result["reconfigurable_success"]},
        ]
    write_csv(outdir / "experiment1_job_size.csv", rows1)
    plot_lines(outdir / "experiment1_job_size.png", rows1, "job_size", "Required cubes", "Experiment 1: Job size")

    rows2 = []
    for failure in [0.0, 0.005, 0.01, 0.02, 0.05]:
        result = estimate(n, 32, failure, 0.0, "random", trials, seed + round(failure * 10000))
        rows2 += [
            {"failure_rate": failure, "series": "Static", "success": result["static_success"]},
            {"failure_rate": failure, "series": "Reconfigurable", "success": result["reconfigurable_success"]},
        ]
    write_csv(outdir / "experiment2_failures.csv", rows2)
    plot_lines(outdir / "experiment2_failures.png", rows2, "failure_rate", "Per-cube failure probability", "Experiment 2: Failures")

    rows3 = []
    for occupied in [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]:
        for pattern in ["random", "fragmented", "concentrated"]:
            result = estimate(n, 32, 0.0, occupied, pattern, trials, seed + round(occupied * 1000) + len(pattern))
            rows3 += [
                {"occupied_rate": occupied, "series": f"Static ({pattern})", "success": result["static_success"]},
                {"occupied_rate": occupied, "series": f"Reconfigurable ({pattern})", "success": result["reconfigurable_success"]},
            ]
    write_csv(outdir / "experiment3_fragmentation.csv", rows3)
    # Use a separate compact plot for the clearest comparison.
    compact = [r for r in rows3 if r["series"] in {"Static (fragmented)", "Static (concentrated)", "Reconfigurable (fragmented)"}]
    plot_lines(outdir / "experiment3_fragmentation.png", compact, "occupied_rate", "Occupied-cube fraction", "Experiment 3: Fragmentation")

    summary = {"experiment1": rows1, "experiment2": rows2, "experiment3": rows3}
    (outdir / "results.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="results", help="Output directory")
    parser.add_argument("--trials", type=int, default=5000, help="Monte Carlo trials per point")
    parser.add_argument("--seed", type=int, default=24, help="Random seed")
    args = parser.parse_args()
    run_all(Path(args.outdir), args.trials, args.seed)
    print(f"Wrote experiment results and graphs to {args.outdir}")


if __name__ == "__main__":
    main()
