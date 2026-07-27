import csv
import os
import random
from datetime import datetime
from multiprocessing import Pool
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx

from helper import xi_a, xi_d

# -----------------------------
# Parameters (edit these)
# -----------------------------
RESULTS_DIR = "models/diffusion/results"
N_RUNS = 1000
N_WORKERS = os.cpu_count()  # multiprocessing pool size for the simulations

# Point this at an existing diffusion_reach.csv to skip the simulation and only
# regenerate plots from that dataset (e.g. "models/diffusion/results/.../diffusion_reach.csv").
INPUT_CSV = None

TARGETING_STRATEGIES = ["degree", "random"]  # betweenness dropped: results track degree closely

N = 100
BASE_SEED = 42

KINDS = ["random", "scale_free", "small_world"]
WEIGHT_TYPES = ["uniform", "degree", "inv_degree"]

# Network params
TARGET_DEGREE = 6  # mean degree for the random (Erdos-Renyi) graph; p is derived from n so density doesn't scale with N
M = 3             # Barabasi-Albert m (scale_free)
K = 6             # Watts-Strogatz k even (small_world)
BETA = 0.1        # Watts-Strogatz rewiring prob (small_world)

# Seeding
SEED_FRAC = 0.03

# Diffusion params
MAX_STEPS = 50

# Threshold distribution (per node)
# Normal distribution clipped to [0, 1]
THETA_MEAN = 0.45
THETA_SD = 0.10

# Equity metric configuration — each value in these lists produces its own
# xi_d_<cutoff> / xi_a_<alpha> column, so add values here to sweep them.
CUTOFFS = [1]
ALPHAS = [1.5]


# -----------------------------
# Network generation
# -----------------------------
def make_network(n: int, kind: str, rng_seed: int, target_degree: float, m: int, k: int, beta: float) -> nx.Graph:
    if kind == "random":
        p = target_degree / (n - 1)
        for attempt in range(100):
            G = nx.erdos_renyi_graph(n, p, seed=rng_seed + attempt)
            if nx.is_connected(G):
                return G
        raise RuntimeError(f"Could not generate a connected Erdős-Rényi graph after 100 attempts (n={n}, p={p})")
    if kind == "scale_free":
        if m < 1 or m >= n:
            raise ValueError("For scale_free, require 1 <= m < n")
        return nx.barabasi_albert_graph(n, m, seed=rng_seed)
    if kind == "small_world":
        if k < 2 or k >= n:
            raise ValueError("For small_world, require 2 <= k < n")
        if k % 2 != 0:
            raise ValueError("For small_world, k must be even in watts_strogatz_graph")
        return nx.watts_strogatz_graph(n, k, beta, seed=rng_seed)

    raise ValueError("Unknown kind. Use: random, scale_free, small_world")


# -----------------------------
# Seeding
# -----------------------------
def seed_by_degree(G: nx.Graph, fraction: float) -> List[int]:
    if not (0.0 <= fraction <= 1.0):
        raise ValueError("seed_frac must be in [0, 1]")

    n = G.number_of_nodes()
    k = int(round(fraction * n))
    if k <= 0:
        return []

    dc = nx.degree_centrality(G)
    return [node for node, _ in sorted(dc.items(), key=lambda x: (x[1], x[0]), reverse=True)[:k]]


def seed_by_betweenness(G: nx.Graph, fraction: float) -> List[int]:
    if not (0.0 <= fraction <= 1.0):
        raise ValueError("seed_frac must be in [0, 1]")

    n = G.number_of_nodes()
    k = int(round(fraction * n))
    if k <= 0:
        return []

    bc = nx.betweenness_centrality(G)
    return [node for node, _ in sorted(bc.items(), key=lambda x: (x[1], x[0]), reverse=True)[:k]]


def seed_by_random(G: nx.Graph, fraction: float, rng: random.Random) -> List[int]:
    if not (0.0 <= fraction <= 1.0):
        raise ValueError("seed_frac must be in [0, 1]")

    n = G.number_of_nodes()
    k = int(round(fraction * n))
    if k <= 0:
        return []

    return rng.sample(list(G.nodes()), k)


def select_seeds(G: nx.Graph, targeting: str, fraction: float, rng: random.Random) -> List[int]:
    if targeting == "degree":
        return seed_by_degree(G, fraction)
    if targeting == "betweenness":
        return seed_by_betweenness(G, fraction)
    if targeting == "random":
        return seed_by_random(G, fraction, rng)
    raise ValueError(f"Unknown targeting: {targeting}")


# -----------------------------
# Threshold sampling
# -----------------------------
def _clip01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def sample_thresholds_normal(G: nx.Graph, rng: random.Random, mean: float, sd: float) -> Dict[int, float]:
    if sd < 0.0:
        raise ValueError("THETA_SD must be >= 0")
    theta_i: Dict[int, float] = {}
    for i in G.nodes():
        t = rng.normalvariate(mean, sd)
        theta_i[i] = _clip01(t)
    return theta_i


# -----------------------------
# Diffusion (synchronous threshold model, heterogeneous theta_i)
# -----------------------------
def diffuse(G: nx.Graph, seeds: List[int], theta_i: Dict[int, float], max_steps: int) -> List[Dict[int, bool]]:
    adopted: Dict[int, bool] = {i: False for i in G.nodes()}
    for i in seeds:
        if i in adopted:
            adopted[i] = True

    history: List[Dict[int, bool]] = [adopted.copy()]

    for _ in range(max_steps):
        new = adopted.copy()

        for i in G.nodes():
            if adopted[i]:
                continue
            nbrs = list(G.neighbors(i))
            if not nbrs:
                continue

            frac = sum(1 for j in nbrs if adopted[j]) / len(nbrs)
            theta = theta_i.get(i, 0.0)
            if frac >= theta:
                new[i] = True

        if new == adopted:
            break

        adopted = new
        history.append(adopted.copy())

    return history


# -----------------------------
# Equity weight schemes (per-node priority weights, used by xi_a and xi_d)
# -----------------------------
def weights_uniform(G: nx.Graph) -> Dict[int, float]:
    return {node: 1.0 for node in G.nodes()}


def _minmax(values: Dict[int, float]) -> Dict[int, float]:
    lo, hi = min(values.values()), max(values.values())
    if hi == lo:
        return {node: 1.0 for node in values}
    return {node: (v - lo) / (hi - lo) for node, v in values.items()}


def weights_degree(G: nx.Graph) -> Dict[int, float]:
    """Higher weight for high-degree (central) nodes, min-max normalised."""
    return _minmax(dict(nx.degree_centrality(G)))


def weights_inv_degree(G: nx.Graph) -> Dict[int, float]:
    """Higher weight for peripheral (low-degree) nodes, min-max normalised."""
    dc = nx.degree_centrality(G)
    return _minmax({node: 1.0 - c for node, c in dc.items()})


def get_weights(G: nx.Graph, weight_type: str) -> Dict[int, float]:
    if weight_type == "uniform":
        return weights_uniform(G)
    if weight_type == "degree":
        return weights_degree(G)
    if weight_type == "inv_degree":
        return weights_inv_degree(G)
    raise ValueError(f"Unknown weight_type: {weight_type}")

# -----------------------------
# Reach
# -----------------------------
def reach(history: List[Dict[int, bool]]) -> Tuple[float, int]:
    final = history[-1]
    n = len(final)
    if n == 0:
        return 0.0, 0
    final_adoption = sum(1 for v in final.values() if v) / n
    steps_to_converge = len(history) - 1
    return final_adoption, steps_to_converge


def run_once(
    kind: str,
    weight_type: str,
    targeting: str,
    run_id: int,
    rng_seed: int,
    theta_mean: float = THETA_MEAN,
    theta_sd: float = THETA_SD,
) -> Dict[str, object]:
    G = make_network(
        n=N,
        kind=kind,
        rng_seed=rng_seed,
        target_degree=TARGET_DEGREE,
        m=M,
        k=K,
        beta=BETA,
    )

    rng = random.Random(rng_seed)
    theta_i = sample_thresholds_normal(G, rng=rng, mean=theta_mean, sd=theta_sd)
    seeds = select_seeds(G, targeting, SEED_FRAC, rng)

    history = diffuse(G, seeds, theta_i=theta_i, max_steps=MAX_STEPS)
    final = history[-1]
    final_adoption, steps_to_converge = reach(history)

    w = get_weights(G, weight_type)
    # xi_a and xi_d share the same resource mapping: R[node] = 1.0 if the
    # node adopted by the end of the run, else 0.0. xi_a uses it continuously
    # (attenuated by distance); xi_d reads it as binary (R[node] > 0).
    R = {node: 1.0 if adopted else 0.0 for node, adopted in final.items()}

    result: Dict[str, object] = {
        "run_id": run_id,
        "kind": kind,
        "weight_type": weight_type,
        "targeting": targeting,
        "rng_seed": rng_seed,
        "n": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "seed_frac": SEED_FRAC,
        "seed_count": len(seeds),
        "theta_mean": theta_mean,
        "theta_sd": theta_sd,
        "max_steps": MAX_STEPS,
        "final_adoption": final_adoption,
        "steps_to_converge": steps_to_converge,
    }

    for cutoff in CUTOFFS:
        xi_d_value, _ = xi_d(G, R, w=w, d=cutoff)
        result[f"xi_d_{cutoff}"] = xi_d_value

    for alpha in ALPHAS:
        xi_a_value, _ = xi_a(G, R, w=w, alpha=alpha)
        result[f"xi_a_{alpha}"] = xi_a_value

    return result


def _run_once_task(task: Tuple[str, str, str, int, int]) -> Dict[str, object]:
    kind, weight_type, targeting, run_id, rng_seed = task
    return run_once(kind, weight_type=weight_type, targeting=targeting, run_id=run_id, rng_seed=rng_seed)


WEIGHT_LABELS = {
    "uniform": "Equal",
    "degree": "Central",
    "inv_degree": "Peripheral",
}
KIND_LABELS = {
    "random": "Random",
    "scale_free": "Scale-free",
    "small_world": "Small-world",
}
TARGETING_LABELS = {
    "degree": "Degree",
    "betweenness": "Betweenness",
    "random": "Random",
}
WEIGHT_COLORS = {
    "uniform": "#4C72B0",
    "degree": "#DD8452",
    "inv_degree": "#55A868",
}

FONTSIZE = 12

# Grouped boxplot layout: one group per targeting strategy, one offset box per weighting scheme.
_BOX_WIDTH = 0.22
_WEIGHT_OFFSETS = {"degree": -0.26, "uniform": 0.0, "inv_degree": 0.26}

_WT_ORDER = ["degree", "uniform", "inv_degree"]


def plot_by_weight_type(rows: List[Dict[str, object]], metric_key: str, metric_label: str, out_path: str) -> None:
    """1x3 grid (kind columns). Targeting strategies on x-axis, weighting schemes as offset boxes."""
    group_centers = list(range(1, len(TARGETING_STRATEGIES) + 1))

    fig, axes = plt.subplots(1, 3, figsize=(8, 3.5), sharey=True)

    for ax, kind in zip(axes, KINDS):
        for wt in _WT_ORDER:
            offset = _WEIGHT_OFFSETS[wt]
            positions = [c + offset for c in group_centers]
            data = [
                [r[metric_key] for r in rows if r["kind"] == kind and r["weight_type"] == wt and r["targeting"] == targeting]
                for targeting in TARGETING_STRATEGIES
            ]
            color = WEIGHT_COLORS[wt]
            bp = ax.boxplot(
                data,
                positions=positions,
                widths=_BOX_WIDTH,
                patch_artist=True,
                medianprops=dict(color="black", linewidth=1.5),
                manage_ticks=False,
            )
            for patch in bp["boxes"]:
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

        ax.set_title(KIND_LABELS[kind], fontsize=FONTSIZE)
        ax.set_xticks(group_centers)
        ax.set_xticklabels([TARGETING_LABELS[t] for t in TARGETING_STRATEGIES], fontsize=FONTSIZE - 2,
                           rotation=20, ha="right")
        ax.set_xlim(group_centers[0] - 0.7, group_centers[-1] + 0.7)
        ax.tick_params(axis="y", labelsize=FONTSIZE)

    axes[0].set_ylabel(metric_label, fontsize=FONTSIZE)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=WEIGHT_COLORS[wt], alpha=0.7, edgecolor="black")
        for wt in _WT_ORDER
    ]
    fig.legend(legend_handles, [WEIGHT_LABELS[wt] for wt in _WT_ORDER],
               loc="lower center", ncol=3, fontsize=FONTSIZE, frameon=False,
               bbox_to_anchor=(0.5, -0.08))

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Plot saved: {out_path}")


def plot_adoption_scatter_grid(rows: List[Dict[str, object]], metric_key: str, metric_label: str, out_path: str) -> None:
    """Small-multiples grid: rows = targeting strategy, columns = kind. Each panel scatters
    adoption vs. metric for individual runs, colored by weighting scheme."""
    n_rows = len(TARGETING_STRATEGIES)
    n_cols = len(KINDS)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8, 2.4 * n_rows), sharex=True, sharey=True)

    for row_i, targeting in enumerate(TARGETING_STRATEGIES):
        for col_i, kind in enumerate(KINDS):
            ax = axes[row_i][col_i]
            for wt in _WT_ORDER:
                subset = [
                    r for r in rows
                    if r["kind"] == kind and r["weight_type"] == wt and r["targeting"] == targeting
                ]
                ax.scatter([r["final_adoption"] for r in subset], [r[metric_key] for r in subset],
                           color=WEIGHT_COLORS[wt], alpha=0.5, s=16, linewidths=0)

            ax.set_xlim(left=-0.02, right=1.02)
            ax.tick_params(labelsize=FONTSIZE - 1)

            if row_i == 0:
                ax.set_title(KIND_LABELS[kind], fontsize=FONTSIZE)
            if row_i == n_rows - 1:
                ax.set_xlabel("Adoption", fontsize=FONTSIZE)
            if col_i == 0:
                ax.set_ylabel(TARGETING_LABELS[targeting], fontsize=FONTSIZE)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=WEIGHT_COLORS[wt], alpha=0.7, edgecolor="black")
        for wt in _WT_ORDER
    ]
    fig.legend(legend_handles, [WEIGHT_LABELS[wt] for wt in _WT_ORDER],
               loc="lower center", ncol=3, fontsize=FONTSIZE, frameon=False,
               bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=(0.03, 0.03, 1, 1))
    fig.text(0.0, 0.5, metric_label, rotation=90, va="center", ha="left", fontsize=FONTSIZE)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Plot saved: {out_path}")


def write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        raise ValueError("No rows to write")

    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def make_output_path(results_dir: str, filename: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(results_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    return os.path.join(run_dir, filename)


def load_rows(path: str) -> List[Dict[str, object]]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = []
        for row in csv.DictReader(f):
            parsed: Dict[str, object] = {}
            for k, v in row.items():
                try:
                    parsed[k] = float(v)
                except ValueError:
                    parsed[k] = v
            rows.append(parsed)
    return rows


def main() -> None:
    if INPUT_CSV:
        print(f"Loading existing dataset: {INPUT_CSV}")
        rows = load_rows(INPUT_CSV)
        out_path = INPUT_CSV
    else:
        tasks: List[Tuple[str, str, str, int, int]] = []
        run_id = 0

        for targeting in TARGETING_STRATEGIES:
            for weight_type in WEIGHT_TYPES:
                for kind in KINDS:
                    for _ in range(N_RUNS):
                        run_id += 1
                        rng_seed = BASE_SEED + (run_id * 1009)  # deterministic, spaced out
                        tasks.append((kind, weight_type, targeting, run_id, rng_seed))

        print(f"Running {len(tasks)} simulations across {N_WORKERS} workers...")
        rows = []
        with Pool(N_WORKERS) as pool:
            for i, result in enumerate(pool.imap_unordered(_run_once_task, tasks, chunksize=20), 1):
                rows.append(result)
                if i % 500 == 0 or i == len(tasks):
                    print(f"  Completed {i}/{len(tasks)} runs")

        out_path = make_output_path(RESULTS_DIR, "diffusion_reach.csv")
        write_csv(out_path, rows)
        print(f"Saved: {out_path}")

    for cutoff in CUTOFFS:
        key = f"xi_d_{cutoff}"
        label = fr"Network inequity ($\xi_d$, $d$={cutoff})"
        plot_by_weight_type(rows, key, label, out_path.replace(".csv", f"_{key}_box.png"))
        plot_adoption_scatter_grid(rows, key, label, out_path.replace(".csv", f"_{key}_scatter.png"))

    for alpha in ALPHAS:
        key = f"xi_a_{alpha}"
        label = fr"Network inequity ($\xi_a$, $\alpha$={alpha})"
        plot_by_weight_type(rows, key, label, out_path.replace(".csv", f"_{key}_box.png"))
        plot_adoption_scatter_grid(rows, key, label, out_path.replace(".csv", f"_{key}_scatter.png"))


if __name__ == "__main__":
    main()
