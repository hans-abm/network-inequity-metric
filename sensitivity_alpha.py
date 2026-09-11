"""
Sensitivity analysis for the distance-decay parameter alpha.

The paper reports the adoption <-> network-inequity relationship at a single
alpha (1.5). This script asks whether that relationship survives a different
choice, and produces one figure answering it.

Why a dedicated script rather than an extra column in main.py: alpha never
affects the diffusion, only the post-hoc metric. run_once computes the adoption
outcome first and only then feeds the resulting R to xi_a. So the whole sweep is
post-processing on *identical* runs -- the curves across alpha share the same
graphs, seeds and adoption outcomes, and carry no Monte Carlo noise between alpha
values. xi_a_multi exploits this by computing all-pairs distances once and
evaluating every alpha against them, so a 9-alpha sweep costs about what a
single-alpha run costs.

The figure has two rows because alpha degenerates at both ends of its range:

  - Low alpha: the min(access, 1.0) cap in xi_a saturates every node, so xi_a
    collapses to 0 with no variance and the regression is undefined.
  - High alpha: access decays to the node's own resource, so xi_a becomes a
    deterministic affine function of adoption (corr -> -1) and any marginal
    effect estimated there is mechanical rather than substantive.

The top row carries the estimate; the bottom row carries sd(xi_a), which falls
away towards both ends as the metric loses its ability to discriminate. The
sweep starts at alpha=0.5 because alpha=0.25 is fully degenerate.

Node weighting is held at WEIGHT_TYPE ("uniform"): the weighting schemes are a
separate question from alpha, and fixing them keeps this to one comparison. The
per-stratum diagnostics that motivated the range (sat_frac, corr_adoption_xi)
are still written to alpha_ame.csv.

Run: python sensitivity_alpha.py
"""

import csv
import os
from multiprocessing import Pool
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import norm

from helper import xi_a_multi
from main import (
    BASE_SEED,
    BETA,
    K,
    KINDS,
    M,
    MAX_STEPS,
    N,
    N_RUNS,
    N_WORKERS,
    SEED_FRAC,
    TARGET_DEGREE,
    TARGETING_STRATEGIES,
    THETA_MEAN,
    THETA_SD,
    WEIGHT_TYPES,
    diffuse,
    get_weights,
    load_rows,
    make_network,
    make_output_path,
    reach,
    sample_thresholds_normal,
    select_seeds,
    write_csv,
)

# -----------------------------
# Parameters (edit these)
# -----------------------------
# Starts at 0.5: at alpha=0.25 the access cap saturates every node, so xi_a is
# constant, the regression is undefined and there is nothing to plot.
ALPHAS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]

# Equal node weighting only -- the weighting schemes are a separate question from
# alpha, and holding this fixed keeps the figure to one comparison.
WEIGHT_TYPE = "uniform"

RESULTS_DIR = "results"
PAPER_DIR = os.path.join("results", "paper")

REFERENCE_ALPHA = 1.5  # the paper's headline value, marked in every panel

# Point this at an existing alpha_sweep_raw.csv to skip the simulation and only
# redo the regression and the plot (mirrors main.py's INPUT_CSV switch).
INPUT_CSV = None

Z_975 = norm.ppf(0.975)
MIN_XI_SD = 1e-9  # below this the design is collinear and statsmodels raises

# -----------------------------
# Labels and style (matching main.py)
# -----------------------------
KIND_LABELS = {"random": "Random", "scale_free": "Scale-free", "small_world": "Small-world"}
TARGETING_LABELS = {"degree": "Degree", "random": "Random"}

FONTSIZE = 12
# With the weighting scheme held fixed, colour is free to carry the seeding
# strategy -- the only comparison left in the figure. Palette from main.py.
_TARGETING_COLORS = {"degree": "#4C72B0", "random": "#DD8452"}

# Every alpha gets a tick; only these get a text label (the rest collide on a log axis).
_LABELLED_ALPHAS = {0.5, 1.0, 1.5, 2.0, 3.0}


def alpha_col(alpha: float) -> str:
    """Metric column name for an alpha. Matches main.py's f"xi_a_{alpha}" so that
    alpha=1.5 yields "xi_a_1.5" and existing datasets stay readable."""
    return f"xi_a_{alpha}"


def satfrac_col(alpha: float) -> str:
    return f"satfrac_a_{alpha}"


# -----------------------------
# Simulation
# -----------------------------
def run_once_alpha(
    kind: str,
    weight_type: str,
    targeting: str,
    run_id: int,
    rng_seed: int,
    theta_mean: float = THETA_MEAN,
    theta_sd: float = THETA_SD,
) -> Dict[str, object]:
    """One simulation run, scored at every alpha in ALPHAS.

    Identical to main.py's run_once through the diffusion -- same graph, seeds,
    thresholds and adoption outcome for a given rng_seed -- then swaps the
    per-alpha xi_a loop for a single xi_a_multi call and additionally records the
    access-cap saturation fraction that the diagnostic row needs.
    """
    import random

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

    # One pass over the shared distance matrix for every alpha.
    scored = xi_a_multi(G, R, w=w, alphas=ALPHAS)
    for alpha in ALPHAS:
        xi, sat_frac = scored[float(alpha)]
        result[alpha_col(alpha)] = xi
        result[satfrac_col(alpha)] = sat_frac

    return result


def _run_once_alpha_task(task: Tuple[str, str, str, int, int]) -> Dict[str, object]:
    kind, weight_type, targeting, run_id, rng_seed = task
    return run_once_alpha(kind, weight_type=weight_type, targeting=targeting,
                          run_id=run_id, rng_seed=rng_seed)


def simulate() -> List[Dict[str, object]]:
    """Run the full n=100 design. Seed derivation copied from main.py so a given
    run_id produces the same graph and adoption outcome as the existing dataset."""
    tasks: List[Tuple[str, str, str, int, int]] = []
    run_id = 0

    # run_id advances over every weight_type, including the ones we skip, so that a
    # given (kind, weight_type, targeting, nth run) keeps the seed -- and therefore
    # the graph and adoption outcome -- it has in main.py's dataset.
    for targeting in TARGETING_STRATEGIES:
        for weight_type in WEIGHT_TYPES:
            for kind in KINDS:
                for _ in range(N_RUNS):
                    run_id += 1
                    rng_seed = BASE_SEED + (run_id * 1009)  # deterministic, spaced out
                    if weight_type != WEIGHT_TYPE:
                        continue
                    tasks.append((kind, weight_type, targeting, run_id, rng_seed))

    print(f"Running {len(tasks)} simulations x {len(ALPHAS)} alphas across {N_WORKERS} workers...")
    rows: List[Dict[str, object]] = []
    with Pool(N_WORKERS) as pool:
        for i, result in enumerate(pool.imap_unordered(_run_once_alpha_task, tasks, chunksize=20), 1):
            rows.append(result)
            if i % 500 == 0 or i == len(tasks):
                print(f"  Completed {i}/{len(tasks)} runs")

    return rows


# -----------------------------
# Estimation
# -----------------------------
def prepare_kinds(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], str]:
    """Make `kind` a categorical with "random" as the reference level."""
    df = df.copy()
    df["kind"] = df["kind"].astype("category")
    ref = "random" if "random" in df["kind"].cat.categories else df["kind"].cat.categories[0]
    df["kind"] = df["kind"].cat.reorder_categories(
        [ref] + [k for k in df["kind"].cat.categories if k != ref], ordered=True
    )
    return df, list(df["kind"].cat.categories), ref


def group_ame_rows(model, alpha, weight_type, targeting, x_col, kinds, ref, df, metric_col):
    """Per-kind average marginal effect of xi_a on adoption, by the delta method.

    Adapted from slopes.py's function of the same name; the additions here are the
    `alpha` column and the standardised effect. Standardising matters because
    sd(xi_a) varies roughly 30-fold across the alpha range -- a raw AME curve would
    move mostly because the units of xi_a change, not because the relationship does.
    """
    params = model.params
    pvalues = model.pvalues
    try:
        cov = model.cov_params()
    except (ValueError, np.linalg.LinAlgError):
        cov = None

    rows = []
    for k in kinds:
        sub = df[df["kind"] == k]
        b_x = params.get(x_col, np.nan)
        if k == ref:
            b_k = b_x
            b_diff = 0.0
            p_diff = np.nan
            var_bk = float(cov.loc[x_col, x_col]) if cov is not None else np.nan
        else:
            inter_term = f"{x_col}:C(kind)[T.{k}]"
            b_diff = params.get(inter_term, np.nan)
            p_diff = pvalues.get(inter_term, np.nan)
            b_k = b_x + (0.0 if np.isnan(b_diff) else b_diff)
            var_bk = float(
                cov.loc[x_col, x_col]
                + cov.loc[inter_term, inter_term]
                + 2 * cov.loc[x_col, inter_term]
            ) if cov is not None else np.nan

        mu = model.predict(sub)
        W_k = float(np.mean(mu * (1.0 - mu)))
        ame = W_k * b_k

        if not np.isnan(var_bk):
            se_ame = abs(W_k) * np.sqrt(max(var_bk, 0.0))
            p_ame = float(2 * norm.sf(abs(ame / se_ame))) if se_ame > 0 else np.nan
            ci_lo = ame - Z_975 * se_ame
            ci_hi = ame + Z_975 * se_ame
        else:
            se_ame = ci_lo = ci_hi = p_ame = np.nan

        # Diagnostics for the bottom row, computed on this stratum only.
        xi_vals = sub[metric_col]
        xi_sd = float(xi_vals.std())
        sat_frac = float(sub[satfrac_col(alpha)].mean()) if satfrac_col(alpha) in sub else np.nan
        if xi_sd > MIN_XI_SD and sub["final_adoption"].std() > MIN_XI_SD:
            corr = float(np.corrcoef(sub["final_adoption"], xi_vals)[0, 1])
        else:
            corr = np.nan

        rows.append({
            "alpha": alpha,
            "targeting": targeting,
            "weight_type": weight_type,
            "kind": k,
            "ame": ame,
            "ame_se": se_ame,
            "ame_ci_lo": ci_lo,
            "ame_ci_hi": ci_hi,
            "ame_p": p_ame,
            # Effect of a 1 SD shift in xi_a -- comparable across alpha.
            "ame_std": ame * xi_sd,
            "ame_std_ci_lo": ci_lo * xi_sd,
            "ame_std_ci_hi": ci_hi * xi_sd,
            "xi_mean": float(xi_vals.mean()),
            "xi_sd": xi_sd,
            "sat_frac": sat_frac,
            "corr_adoption_xi": corr,
            "interaction_coef": b_diff,
            "interaction_p": p_diff,
            "n_obs": int(len(sub)),
        })
    return rows


def degenerate_rows(alpha, weight_type, targeting, kinds, df, metric_col):
    """Placeholder rows for a stratum where xi_a has no variance (alpha too low).

    The regression cannot be fitted there, but the diagnostic row still needs the
    saturation fraction -- that is precisely the evidence the panel is shading.
    """
    rows = []
    for k in kinds:
        sub = df[df["kind"] == k]
        rows.append({
            "alpha": alpha, "targeting": targeting, "weight_type": weight_type, "kind": k,
            "ame": np.nan, "ame_se": np.nan, "ame_ci_lo": np.nan, "ame_ci_hi": np.nan,
            "ame_p": np.nan, "ame_std": np.nan, "ame_std_ci_lo": np.nan, "ame_std_ci_hi": np.nan,
            "xi_mean": float(sub[metric_col].mean()),
            "xi_sd": float(sub[metric_col].std()),
            "sat_frac": float(sub[satfrac_col(alpha)].mean()) if satfrac_col(alpha) in sub else np.nan,
            "corr_adoption_xi": np.nan,
            "interaction_coef": np.nan, "interaction_p": np.nan,
            "n_obs": int(len(sub)),
        })
    return rows


def estimate(df: pd.DataFrame) -> pd.DataFrame:
    """Fractional logit AMEs at every alpha, stratified by weight_type x targeting."""
    ame_rows: List[Dict[str, object]] = []

    for alpha in ALPHAS:
        metric_col = alpha_col(alpha)
        if metric_col not in df.columns:
            print(f"  alpha={alpha}: column {metric_col} missing, skipped")
            continue

        df_kinds, kinds, ref = prepare_kinds(df)
        # patsy chokes on the literal "." in e.g. xi_a_1.5, so quote the term.
        term = f'Q("{metric_col}")'

        n_degenerate = 0
        for (wt, targeting), grp in df_kinds.groupby(["weight_type", "targeting"], observed=True):
            if grp[metric_col].std() <= MIN_XI_SD:
                # xi_a is constant here: the access cap has saturated everything.
                ame_rows.extend(degenerate_rows(alpha, wt, targeting, kinds, grp, metric_col))
                n_degenerate += 1
                continue

            formula = f"final_adoption ~ {term} * C(kind)"
            try:
                model = smf.glm(formula, data=grp, family=sm.families.Binomial()).fit(cov_type="HC3")
            except Exception as exc:
                print(f"  alpha={alpha} {wt}/{targeting}: fit failed ({exc}), emitting NaN")
                ame_rows.extend(degenerate_rows(alpha, wt, targeting, kinds, grp, metric_col))
                n_degenerate += 1
                continue

            ame_rows.extend(group_ame_rows(
                model, alpha, wt, targeting, term, kinds, ref, grp, metric_col
            ))

        note = f"  ({n_degenerate} strata degenerate)" if n_degenerate else ""
        print(f"  alpha={alpha}: done{note}")

    return pd.DataFrame(ame_rows).sort_values(["alpha", "targeting", "weight_type", "kind"])


# -----------------------------
# The figure
# -----------------------------
def plot_alpha_sensitivity(ame_df: pd.DataFrame, out_paths: List[str]) -> None:
    """One figure: standardised AME vs alpha, over sd(xi_a).

    Columns are network kinds, colour is the seeding strategy, node weighting is
    held at WEIGHT_TYPE. The top row is the estimate; the bottom row is sd(xi_a),
    which falls away at both ends of the alpha range as the metric loses its
    ability to discriminate. Both rows share a y-axis across kinds so the panels
    can be read against each other.
    """
    n_cols = len(KINDS)
    fig, axes = plt.subplots(2, n_cols, figsize=(8, 4.6), sharex=True, sharey="row")

    for col_i, kind in enumerate(KINDS):
        ax_top = axes[0][col_i]
        ax_bot = axes[1][col_i]
        panel = ame_df[ame_df["kind"] == kind]

        for ax in (ax_top, ax_bot):
            ax.axvline(REFERENCE_ALPHA, ls=":", color="black", lw=0.9, zorder=1)
            ax.set_xscale("log")
            ax.tick_params(labelsize=FONTSIZE - 2)

        for targeting in TARGETING_STRATEGIES:
            sub = panel[panel["targeting"] == targeting].sort_values("alpha")
            if sub.empty:
                continue

            color = _TARGETING_COLORS.get(targeting, "#4C72B0")
            ax_top.plot(sub["alpha"], sub["ame_std"], color=color, lw=1.8, zorder=3)
            ax_top.fill_between(sub["alpha"], sub["ame_std_ci_lo"], sub["ame_std_ci_hi"],
                                color=color, alpha=0.2, linewidth=0, zorder=2)
            ax_bot.plot(sub["alpha"], sub["xi_sd"], color=color, lw=1.8, zorder=3)

        ax_top.set_title(KIND_LABELS[kind], fontsize=FONTSIZE)
        ax_bot.set_xlabel(r"$\alpha$ (distance decay)", fontsize=FONTSIZE - 1)
        # Tick every alpha, but label only a subset -- on a log axis the full set
        # collides around 0.75-1.5.
        ax_bot.set_xticks(ALPHAS)
        ax_bot.set_xticklabels(
            [f"{a:g}" if a in _LABELLED_ALPHAS else "" for a in ALPHAS],
            fontsize=FONTSIZE - 3,
        )
        ax_bot.minorticks_off()

        if col_i == 0:
            ax_top.set_ylabel(r"AME per 1 SD of $\xi_a$", fontsize=FONTSIZE - 1)
            ax_bot.set_ylabel(r"sd($\xi_a$)", fontsize=FONTSIZE - 1)

    legend_handles = [
        plt.Line2D([0], [0], color=_TARGETING_COLORS[t], lw=2) for t in TARGETING_STRATEGIES
    ]
    legend_labels = [f"{TARGETING_LABELS.get(t, t)} seeding" for t in TARGETING_STRATEGIES]
    fig.legend(legend_handles, legend_labels, loc="lower center",
               ncol=2, fontsize=FONTSIZE - 1, frameon=False, bbox_to_anchor=(0.5, -0.06))

    fig.tight_layout()
    for out_path in out_paths:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight", dpi=300)
        print(f"Plot saved: {out_path}")
    plt.close(fig)


# -----------------------------
# Driver
# -----------------------------
def main() -> None:
    if INPUT_CSV:
        print(f"Loading existing dataset: {INPUT_CSV}")
        rows = load_rows(INPUT_CSV)
        raw_path = INPUT_CSV
    else:
        rows = simulate()
        raw_path = make_output_path(f"{RESULTS_DIR}", "alpha_sweep_raw.csv")
        # Stamp the run directory with what it is, matching the other sweeps.
        run_dir = os.path.dirname(raw_path)
        stamped = f"{run_dir}_alpha_sensitivity"
        if not os.path.exists(stamped):
            os.rename(run_dir, stamped)
            raw_path = os.path.join(stamped, "alpha_sweep_raw.csv")
        write_csv(raw_path, rows)
        print(f"Saved: {raw_path}")

    df = pd.DataFrame(rows)

    print(f"\nEstimating AMEs at {len(ALPHAS)} alphas...")
    ame_df = estimate(df)

    ame_path = os.path.join(os.path.dirname(raw_path), "alpha_ame.csv")
    ame_df.round(6).to_csv(ame_path, index=False)
    print(f"Saved: {ame_path}")

    print(f"\n=== Standardised AME (adoption per 1 SD of xi_a), {REFERENCE_ALPHA=} highlighted ===")
    summary = ame_df.pivot_table(
        index=["targeting", "weight_type", "kind"], columns="alpha", values="ame_std"
    )
    print(summary.round(3).to_string())

    plot_alpha_sensitivity(ame_df, [
        os.path.join(os.path.dirname(raw_path), "alpha_sensitivity.png"),
        os.path.join(PAPER_DIR, "alpha_sensitivity.png"),
        os.path.join(PAPER_DIR, "alpha_sensitivity.pdf"),
    ])


if __name__ == "__main__":
    main()
