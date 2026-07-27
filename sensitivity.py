"""
Sensitivity analysis: sweep theta_mean x theta_sd and report
mean/SD of reach, xi_d, and xi_a by network kind.
"""

import csv
import os
import statistics
from datetime import datetime
from itertools import product
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from main import (
    ALPHAS,
    BASE_SEED,
    CUTOFFS,
    KINDS,
    N_RUNS,
    RESULTS_DIR,
    run_once,
)

THETA_MEANS = [0.30, 0.45, 0.60]
THETA_SDS = [0.05, 0.10, 0.15]

# Held fixed while sweeping theta_mean x theta_sd.
WEIGHT_TYPE = "uniform"
TARGETING = "degree"

METRIC_KEYS = ["final_adoption"] + [f"xi_d_{c}" for c in CUTOFFS] + [f"xi_a_{a}" for a in ALPHAS]

KIND_LABELS = {"random": "Random", "scale_free": "Scale-free", "small_world": "Small-world"}


def _metric_label(key: str) -> str:
    if key == "final_adoption":
        return "Mean adoption"
    if key.startswith("xi_d_"):
        return fr"Mean inequity ($\xi_d$, d={key.split('_', 2)[2]})"
    if key.startswith("xi_a_"):
        return fr"Mean inequity ($\xi_a$, $\alpha$={key.split('_', 2)[2]})"
    return key


METRIC_LABELS = {key: _metric_label(key) for key in METRIC_KEYS}
METRIC_CMAPS = {key: ("YlGn" if key == "final_adoption" else "YlOrRd") for key in METRIC_KEYS}


def make_output_path(results_dir: str, filename: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(results_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    return os.path.join(run_dir, filename)


def summarize(batch: List[Dict[str, object]]) -> Dict[str, float]:
    stats: Dict[str, float] = {}
    for key in METRIC_KEYS:
        values = [r[key] for r in batch]
        stats[f"mean_{key}"] = statistics.mean(values)
        stats[f"sd_{key}"] = statistics.stdev(values)
    return stats


def build_matrix(summary: List[Dict[str, object]], kind: str, key: str) -> np.ndarray:
    # rows = theta_sd (low -> high, bottom to top), cols = theta_mean (low -> high)
    lookup = {(s["kind"], s["theta_mean"], s["theta_sd"]): s for s in summary}
    mat = np.zeros((len(THETA_SDS), len(THETA_MEANS)))
    for r, sd in enumerate(THETA_SDS):
        for c, tm in enumerate(THETA_MEANS):
            mat[r, c] = lookup[(kind, tm, sd)][key]
    return mat


def plot_heatmap(summary: List[Dict[str, object]], out_path: str) -> None:
    n_rows = len(METRIC_KEYS)
    n_cols = len(KINDS)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.5 * n_cols, 2.2 * n_rows),
        constrained_layout=True,
        squeeze=False,
    )

    theta_labels = [f"{tm:.2f}" for tm in THETA_MEANS]
    sd_labels = [f"{sd:.2f}" for sd in THETA_SDS]

    for row_i, metric_key in enumerate(METRIC_KEYS):
        mats = {kind: build_matrix(summary, kind, f"mean_{metric_key}") for kind in KINDS}
        sd_mats = {kind: build_matrix(summary, kind, f"sd_{metric_key}") for kind in KINDS}
        vmin, vmax = 0.0, 1.0

        for col_i, kind in enumerate(KINDS):
            ax = axes[row_i][col_i]
            mat = mats[kind]
            sd_mat = sd_mats[kind]

            im = ax.imshow(
                mat,
                aspect="auto",
                origin="lower",
                vmin=vmin, vmax=vmax,
                cmap=METRIC_CMAPS[metric_key],
            )

            for r in range(len(THETA_SDS)):
                for c in range(len(THETA_MEANS)):
                    val = mat[r, c]
                    color = "white" if val > 0.6 else "black"
                    ax.text(c, r, f"{val:.2f}\n±{sd_mat[r, c]:.2f}", ha="center", va="center",
                            fontsize=7, color=color)

            ax.set_xticks(range(len(THETA_MEANS)))
            ax.set_yticks(range(len(THETA_SDS)))

            if row_i == n_rows - 1:
                ax.set_xticklabels(theta_labels, fontsize=8)
                ax.set_xlabel("Threshold mean (θ)", fontsize=8)
            else:
                ax.set_xticklabels([])

            if col_i == 0:
                ax.set_yticklabels(sd_labels, fontsize=8)
                ax.set_ylabel("Threshold SD (σ)", fontsize=8)
            else:
                ax.set_yticklabels([])

            if row_i == 0:
                ax.set_title(KIND_LABELS[kind], fontsize=9, fontweight="bold")

        cbar = fig.colorbar(
            im,
            ax=axes[row_i, :],
            shrink=0.85,
            pad=0.02,
        )
        cbar.set_label(METRIC_LABELS[metric_key], fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> None:
    combos = list(product(KINDS, THETA_MEANS, THETA_SDS))
    total_runs = len(combos) * N_RUNS
    print(f"Running {len(combos)} parameter combinations x {N_RUNS} runs = {total_runs} total\n")

    all_rows: List[Dict[str, object]] = []
    summary: List[Dict[str, object]] = []

    global_run_id = 0
    for kind, theta_mean, theta_sd in combos:
        batch: List[Dict[str, object]] = []
        for _ in range(N_RUNS):
            global_run_id += 1
            rng_seed = BASE_SEED + global_run_id * 1009
            batch.append(run_once(
                kind,
                weight_type=WEIGHT_TYPE,
                targeting=TARGETING,
                run_id=global_run_id,
                rng_seed=rng_seed,
                theta_mean=theta_mean,
                theta_sd=theta_sd,
            ))
        all_rows.extend(batch)

        stats = summarize(batch)
        summary.append({"kind": kind, "theta_mean": theta_mean, "theta_sd": theta_sd, **stats})

        metrics_str = "  ".join(
            f"{key}={stats[f'mean_{key}']:.3f}±{stats[f'sd_{key}']:.3f}" for key in METRIC_KEYS
        )
        print(f"  {kind:12s}  tm={theta_mean:.2f}  sd={theta_sd:.2f}  {metrics_str}")

    raw_path = make_output_path(RESULTS_DIR, "sensitivity_raw.csv")
    fieldnames = list(all_rows[0].keys())
    with open(raw_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    summary_path = raw_path.replace("sensitivity_raw.csv", "sensitivity_summary.csv")
    summary_fields = list(summary[0].keys())
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        for s in summary:
            w.writerow(s)

    plot_path = summary_path.replace("sensitivity_summary.csv", "sensitivity_heatmap.png")
    plot_heatmap(summary, plot_path)

    print(f"\nRaw data : {raw_path}")
    print(f"Summary  : {summary_path}")
    print(f"Heatmap  : {plot_path}")


if __name__ == "__main__":
    main()