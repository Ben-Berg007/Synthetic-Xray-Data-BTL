"""
Step 6: Statistics for Q1 and Q2 from the test-set predictions.

Run from D:\\School\\synthetic_data:   python scripts\\06_stats.py
Reads outputs\\step5\\predictions.csv; writes outputs\\step6\\:
  summary_by_condition.csv   mean, SD, min, max of every metric across seeds
  comparisons.csv            pre-specified paired comparisons of F1 (see below)
  gap_recovered.csv          per seed: BTL - real_full, tier, share of gap recovered (vs real_small and synth_only), recall guard
  gap_verdict.csv            overall pre-registered verdict: CLOSED / LIKELY CLOSED / PARTIALLY CLOSED / NOT CLOSED
  confusion_matrices.csv     per seed and pooled
  f1_by_condition.png        F1 per condition with each seed as a point

Pre-specified comparisons (per instructor feedback):
  Q1   synth_only vs real_full
  Q2a  btl vs real_small        (primary Q2 test)
  Q2b  btl vs synth_only        (primary Q2 test)
  ref  btl vs real_full         (how far BTL remains from the full-data benchmark)
Omnibus first: Friedman test across the four conditions (seeds as blocks).
For each comparison:
  - mean F1 difference and per-seed differences
  - 95% CI from a two-level paired bootstrap (resample seeds, then test images; the same
    resampled images are used for both conditions, so the comparison stays paired)
  - paired t-test across seeds (df = n_seeds - 1)
  - Wilcoxon signed-rank across seeds (10 seeds -> smallest possible two-sided p = 0.002;
    with only 5 seeds it would be 0.0625, which is why the study uses 10)
  - McNemar exact test per seed on correct/incorrect predictions (image level)
  - Holm correction across the three primary tests (Q1, Q2a, Q2b) on the bootstrap p-values
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (f1_score, precision_score, recall_score, accuracy_score,
                             average_precision_score, roc_auc_score, confusion_matrix)

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--file", default=str(ROOT / "outputs" / "step5" / "predictions.csv"))
ap.add_argument("--boot", type=int, default=2000)
ap.add_argument("--threshold", type=float, default=0.5)
ap.add_argument("--margin", type=float, default=0.05, help="pre-registered: 'closed' = BTL within this F1 of real_full")
ap.add_argument("--partial", type=float, default=0.10, help="pre-registered: 'partially closed' = within this F1")
args = ap.parse_args()
OUT = ROOT / "outputs" / "step6"; OUT.mkdir(parents=True, exist_ok=True)
P = pd.read_csv(args.file)
CONDS = [c for c in ["real_full", "real_small", "synth_only", "btl"] if c in P.condition.unique()]
SEEDS = sorted(P.seed.unique())
thr = args.threshold

def f1_fast(y, yh):
    tp = (y & yh).sum(); fp = (~y & yh).sum(); fn = (y & ~yh).sum()
    return 2 * tp / max(2 * tp + fp + fn, 1)

# arrays[seed][cond] -> (y bool, yhat bool, prob)
arr = {}
for (s, c), g in P.groupby(["seed", "condition"]):
    g = g.sort_values("test_idx")
    arr.setdefault(s, {})[c] = (g.y.values.astype(bool), (g.prob.values >= thr), g.prob.values)

# ---------- per seed x condition metrics ----------
rows = []
for s in SEEDS:
    for c in CONDS:
        y, yh, p = arr[s][c]
        tn, fp, fn, tp = confusion_matrix(y, yh, labels=[False, True]).ravel()
        rows.append(dict(seed=s, condition=c, f1=f1_score(y, yh, zero_division=0),
                         precision=precision_score(y, yh, zero_division=0), recall=recall_score(y, yh, zero_division=0),
                         specificity=tn / max(tn + fp, 1), accuracy=accuracy_score(y, yh),
                         pr_auc=average_precision_score(y, p), roc_auc=roc_auc_score(y, p) if y.any() and (~y).any() else np.nan,
                         tn=tn, fp=fp, fn=fn, tp=tp))
M = pd.DataFrame(rows)
metric_cols = ["f1", "precision", "recall", "specificity", "accuracy", "pr_auc", "roc_auc"]
summ = M.groupby("condition")[metric_cols].agg(["mean", "std", "min", "max"]).round(4).reindex(CONDS)
summ.to_csv(OUT / "summary_by_condition.csv")
cm = pd.concat([M[["seed", "condition", "tn", "fp", "fn", "tp"]],
                M.groupby("condition")[["tn", "fp", "fn", "tp"]].sum().reset_index().assign(seed="pooled")])
cm.to_csv(OUT / "confusion_matrices.csv", index=False)

# ---------- omnibus ----------
wide = M.pivot(index="seed", columns="condition", values="f1")[CONDS]
fried = stats.friedmanchisquare(*[wide[c].values for c in CONDS]) if len(CONDS) >= 3 and len(SEEDS) >= 2 else None

# ---------- paired comparisons ----------
rng = np.random.default_rng(2026)
n_test = len(next(iter(arr[SEEDS[0]].values()))[0])
def boot_diff(a, b):
    """two-level paired bootstrap of mean F1(a) - F1(b)"""
    out = np.empty(args.boot)
    for i in range(args.boot):
        ss = rng.choice(SEEDS, size=len(SEEDS), replace=True); d = []
        for s in ss:
            idx = rng.integers(0, n_test, n_test)
            y = arr[s][a][0][idx]
            d.append(f1_fast(y, arr[s][a][1][idx]) - f1_fast(y, arr[s][b][1][idx]))
        out[i] = np.mean(d)
    return out

comps = [("Q1", "synth_only", "real_full"), ("Q2a", "btl", "real_small"),
         ("Q2b", "btl", "synth_only"), ("ref", "btl", "real_full")]
res = []
for name, a, b in comps:
    if a not in CONDS or b not in CONDS:
        continue
    d = (wide[a] - wide[b]).values
    bd = boot_diff(a, b)
    p_boot = min(1.0, 2 * min((bd <= 0).mean(), (bd >= 0).mean()))
    t = stats.ttest_rel(wide[a], wide[b]) if len(SEEDS) > 1 else None
    try:
        w = stats.wilcoxon(d) if len(SEEDS) > 1 and np.any(d != 0) else None
    except ValueError:
        w = None
    mcn = []
    for s in SEEDS:
        y = arr[s][a][0]; ca = arr[s][a][1] == y; cb = arr[s][b][1] == y
        n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
        mcn.append(stats.binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 > 0 else 1.0)
    res.append(dict(comparison=name, a=a, b=b, mean_f1_a=wide[a].mean(), mean_f1_b=wide[b].mean(),
                    mean_diff=d.mean(), per_seed_diff=";".join(f"{x:+.4f}" for x in d),
                    seeds_a_better=int((d > 0).sum()), n_seeds=len(d),
                    ci95_low=np.percentile(bd, 2.5), ci95_high=np.percentile(bd, 97.5), p_bootstrap=p_boot,
                    t_stat=t.statistic if t else np.nan, p_ttest=t.pvalue if t else np.nan,
                    p_wilcoxon=w.pvalue if w else np.nan,
                    mcnemar_p_per_seed=";".join(f"{x:.4f}" for x in mcn),
                    mcnemar_seeds_sig=int(sum(x < 0.05 for x in mcn))))
C = pd.DataFrame(res)
prim = C.comparison.isin(["Q1", "Q2a", "Q2b"])
if prim.any():
    pv = C.loc[prim, "p_bootstrap"].values; order = np.argsort(pv); m = len(pv); adj = np.empty(m); run = 0
    for r, i in enumerate(order):
        run = max(run, min(1, (m - r) * pv[i])); adj[i] = run
    C.loc[prim, "p_bootstrap_holm"] = adj
C.round(5).to_csv(OUT / "comparisons.csv", index=False)

# ---------- gap closure (pre-registered 2026-09-27, before any classifier results) ----------
# d = F1(btl) - F1(real_full)
#   CLOSED            95% CI lower bound of d >= -margin   (non-inferiority shown)
#   LIKELY CLOSED     mean d >= -margin, but CI lower bound < -margin
#   PARTIALLY CLOSED  -partial <= mean d < -margin
#   NOT CLOSED        mean d < -partial
# Guard: flag if BTL recall (sensitivity) is more than `margin` below real_full (more missed pneumonia).
# Relative measure: share of the gap recovered = (btl - base) / (real_full - base), base = real_small and synth_only.
def tier(mean_d, lo):
    if lo >= -args.margin: return "CLOSED"
    if mean_d >= -args.margin: return "LIKELY CLOSED"
    if mean_d >= -args.partial: return "PARTIALLY CLOSED"
    return "NOT CLOSED"
g = None
if {"btl", "real_full"} <= set(CONDS):
    ref = C[C.comparison == "ref"].iloc[0]
    rec = M.pivot(index="seed", columns="condition", values="recall")
    g = pd.DataFrame(index=wide.index)
    g["btl_minus_full"] = wide.btl - wide.real_full
    g["seed_tier"] = [("CLOSED/LIKELY" if x >= -args.margin else "PARTIAL" if x >= -args.partial else "NOT") for x in g.btl_minus_full]
    for base in ("real_small", "synth_only"):
        if base in CONDS:
            gap = wide.real_full - wide[base]
            g[f"recovered_vs_{base}"] = np.where(gap > 0.001, (wide.btl - wide[base]) / gap, np.nan)
    g["recall_btl_minus_full"] = rec.btl - rec.real_full
    g.round(4).to_csv(OUT / "gap_recovered.csv")
    verdict = dict(mean_btl_minus_full=ref.mean_diff, ci95_low=ref.ci95_low, ci95_high=ref.ci95_high,
                   margin=args.margin, partial=args.partial, tier=tier(ref.mean_diff, ref.ci95_low),
                   recall_guard_ok=bool(g.recall_btl_minus_full.mean() >= -args.margin),
                   mean_recovered_vs_real_small=np.nanmean(g.get("recovered_vs_real_small", np.nan)),
                   mean_recovered_vs_synth_only=np.nanmean(g.get("recovered_vs_synth_only", np.nan)))
    pd.Series(verdict).to_csv(OUT / "gap_verdict.csv", header=["value"])

# ---------- figure ----------
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 4), dpi=200)
    labels = {"real_full": "Real-only\n(full pool)", "real_small": "Real-only\n(10–11 images)",
              "synth_only": "Synthetic-only", "btl": "BTL"}
    for i, c in enumerate(CONDS):
        v = wide[c].values
        ax.bar(i, v.mean(), color="#BFBFBF", edgecolor="#333", width=0.6)
        ax.scatter(np.full(len(v), i) + np.linspace(-0.12, 0.12, len(v)), v, color="#222", s=14, zorder=3)
    for s in SEEDS:
        ax.plot(range(len(CONDS)), wide.loc[s, CONDS].values, color="#888", lw=0.6, alpha=0.6, zorder=2)
    ax.set_xticks(range(len(CONDS)), [labels[c] for c in CONDS]); ax.set_ylabel("F1-score (held-out real test set)")
    ax.set_ylim(0, 1); ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("F1 by training condition (bars = mean; points/lines = seeds)", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "f1_by_condition.png")
except Exception as e:
    print("figure skipped:", e)

# ---------- print ----------
pd.set_option("display.width", 200)
print("Mean (SD) across seeds:"); print(summ.xs("mean", axis=1, level=1).round(3).to_string())
print("\nFriedman omnibus:", f"chi2={fried.statistic:.3f}, p={fried.pvalue:.4f}" if fried else "n/a")
print("\nComparisons (F1):")
print(C[["comparison", "a", "b", "mean_diff", "ci95_low", "ci95_high", "p_bootstrap", "p_ttest", "p_wilcoxon",
         "seeds_a_better", "mcnemar_seeds_sig"]].round(4).to_string(index=False))
if g is not None:
    print("\nGap closure per seed:"); print(g.round(3).to_string())
    print("\nVERDICT:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in verdict.items()})
