import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from scipy.stats import norm

# Point this at a main.py-generated diffusion_reach.csv (has both xi_d_* and xi_a_*
# columns, plus a targeting column with "degree" / "random" strategies).
CSV_PATH = "results/20260722_140942_100/diffusion_reach.csv"

df = pd.read_csv(CSV_PATH)

METRICS = [("xi_d", "xi_d_1"), ("xi_a", "xi_a_1.5")]

Z_975 = norm.ppf(0.975)  # 95% CI critical value


def prepare_kinds(df):
    df = df.copy()
    df["kind"] = df["kind"].astype("category")
    ref = "random" if "random" in df["kind"].cat.categories else df["kind"].cat.categories[0]
    df["kind"] = df["kind"].cat.reorder_categories(
        [ref] + [k for k in df["kind"].cat.categories if k != ref], ordered=True
    )
    return df, list(df["kind"].cat.categories), ref


def group_ame_rows(model, metric, weight_type, targeting, x_col, kinds, ref, df):
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
            se_ame = np.nan
            p_ame = np.nan
            ci_lo = np.nan
            ci_hi = np.nan
        rows.append({
            "metric": metric,
            "targeting": targeting,
            "weight_type": weight_type,
            "kind": k,
            "ame": round(ame, 4),
            "ame_se": round(se_ame, 4) if not np.isnan(se_ame) else np.nan,
            "ame_ci_lo": round(ci_lo, 4) if not np.isnan(ci_lo) else np.nan,
            "ame_ci_hi": round(ci_hi, 4) if not np.isnan(ci_hi) else np.nan,
            "ame_p": round(p_ame, 4) if not np.isnan(p_ame) else np.nan,
            "interaction_coef": round(b_diff, 4),
            "interaction_p": round(p_diff, 4) if not np.isnan(p_diff) else np.nan,
        })
    return rows


# Boundary values (fractional logit handles 0/1 natively)
print("Boundary values:")
for metric, x_col in METRICS:
    print(f"  {metric}:")
    for col in ["final_adoption", x_col]:
        n0 = (df[col] == 0).sum()
        n1 = (df[col] == 1).sum()
        print(f"    {col}: {n0} zeros, {n1} ones out of {len(df)}")


# -----------------------------
# 1. Fractional logit AME
#    final_adoption ~ {x_col} * C(kind), stratified by weight_type x targeting
# -----------------------------

ame_rows = []

for metric, x_col in METRICS:
    df_kinds, kinds, ref = prepare_kinds(df)
    term = f'Q("{x_col}")'  # patsy chokes on the literal "." in e.g. xi_a_1.5, so quote it
    for (wt, targeting), grp in df_kinds.groupby(["weight_type", "targeting"]):
        formula = f"final_adoption ~ {term} * C(kind)"
        model = smf.glm(formula, data=grp, family=sm.families.Binomial()).fit(cov_type="HC3")
        ame_rows.extend(group_ame_rows(model, metric, wt, targeting, term, kinds, ref, grp))

ame_df = pd.DataFrame(ame_rows).sort_values(["metric", "targeting", "weight_type", "kind"])
ame_df.to_csv("ame_results.csv", index=False)

print("\n=== Fractional logit AME: final_adoption ~ xi_d/xi_a * C(kind), by targeting x weight_type ===")
print(ame_df.to_string(index=False))


# -----------------------------
# 2. Gap analysis
# -----------------------------
# Gap = mean(metric | weight_type) - mean(metric | weight_type == "uniform"), within each (kind, targeting)

gap_rows = []

for metric, col in METRICS:
    for (kind, targeting), grp in df.groupby(["kind", "targeting"]):
        ref_vals = grp.loc[grp["weight_type"] == "uniform", col]
        ref_mean = ref_vals.mean()
        for wt, sub in grp.groupby("weight_type"):
            vals = sub[col]
            gap = vals.mean() - ref_mean
            if wt == "uniform":
                t, p = np.nan, np.nan
                ci_lo, ci_hi = np.nan, np.nan
            else:
                t, p = stats.ttest_ind(vals, ref_vals, equal_var=False)
                se_diff = np.sqrt(vals.var(ddof=1) / len(vals) + ref_vals.var(ddof=1) / len(ref_vals))
                ci_lo = gap - Z_975 * se_diff
                ci_hi = gap + Z_975 * se_diff
            gap_rows.append({
                "metric": metric, "kind": kind, "targeting": targeting, "weight_type": wt,
                "mean": round(vals.mean(), 4), "sd": round(vals.std(), 4),
                "gap_vs_uniform": round(gap, 4),
                "gap_ci_lo": round(ci_lo, 4) if not np.isnan(ci_lo) else np.nan,
                "gap_ci_hi": round(ci_hi, 4) if not np.isnan(ci_hi) else np.nan,
                "t_stat": round(t, 4) if not np.isnan(t) else np.nan,
                "p_value": round(p, 4) if not np.isnan(p) else np.nan,
            })

gap_df = pd.DataFrame(gap_rows).sort_values(["metric", "targeting", "kind", "weight_type"])
print("\n=== Gap analysis: xi_d / xi_a by (kind, targeting, weight_type) vs uniform baseline ===")
print(gap_df.to_string(index=False))
