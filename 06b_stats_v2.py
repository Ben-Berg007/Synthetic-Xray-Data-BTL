"""
Step 6b (RUN 2): pre-registered analysis (2026-09-30) of outputs\\step5b\\predictions_v2.csv.
    python scripts\\06b_stats_v2.py
Primary comparison metric: MCC at threshold 0.5 (chance = 0; robust to the 74% pneumonia prevalence).
Also reported: F1 (with the always-pneumonia floor of 0.8515 as a triviality check) and ROC-AUC.
Secondary: mcc_cal = MCC at each model's validation-calibrated threshold ("with a small labelled calibration set").
Gap closure tiers (BTL vs real_full) are judged on ROC-AUC: CLOSED if the 95% CI lower bound >= -0.05;
LIKELY CLOSED if mean >= -0.05; PARTIALLY CLOSED if mean >= -0.10; else NOT CLOSED.
Comparisons: Q1 synth_only vs real_full; Q2a btl vs real_small; Q2b btl vs synth_only (Holm across these);
ref btl vs real_full. Per comparison: two-level paired bootstrap 95% CI, paired t, Wilcoxon, McNemar per seed.
Outputs in outputs\\step6b\\.
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--file", default=str(ROOT / "outputs" / "step5b" / "predictions_v2.csv"))
ap.add_argument("--boot", type=int, default=1000)
args = ap.parse_args()
OUT = ROOT / "outputs" / "step6b"; OUT.mkdir(parents=True, exist_ok=True)
P = pd.read_csv(args.file); THR, MARGIN, PARTIAL = 0.5, 0.05, 0.10
CONDS = [c for c in ["real_full", "real_small", "synth_only", "btl"] if c in P.condition.unique()]
SEEDS = sorted(P.seed.unique())
TRIVIAL_F1 = None

def mcc(y, p):
    yh = p >= THR; tp = float((y & yh).sum()); tn = float((~y & ~yh).sum()); fp = float((~y & yh).sum()); fn = float((y & ~yh).sum())
    d = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)); return (tp * tn - fp * fn) / d if d else 0.0
def f1(y, p):
    yh = p >= THR; tp = (y & yh).sum(); return 2 * tp / max(2 * tp + (~y & yh).sum() + (y & ~yh).sum(), 1)
def auc(y, p):
    return roc_auc_score(y, p) if y.any() and (~y).any() else np.nan
METRICS = {"mcc": mcc, "f1": f1, "roc_auc": auc}
SRC = {"mcc": "A", "f1": "A", "roc_auc": "A"}

# Secondary (pre-registered 2026-09-30): "mcc_cal" = MCC at each model's validation-calibrated threshold (thr_val).
# Implemented by rescaling probabilities so that thr_val maps to 0.5; ranking (ROC-AUC) is unchanged.
HAS_CAL = "thr_val" in P.columns
A, A_cal = {}, {}
for (s, c), g in P.groupby(["seed", "condition"]):
    g = g.sort_values("test_idx"); y = g.y.values.astype(bool); p = g.prob.values; A.setdefault(s, {})[c] = (y, p)
    if HAS_CAL:
        t = float(g.thr_val.iloc[0]); pc = np.where(p >= t, 0.5 + 0.5 * (p - t) / max(1 - t, 1e-6), 0.5 * p / max(t, 1e-6))
        A_cal.setdefault(s, {})[c] = (y, pc)
y0 = A[SEEDS[0]][CONDS[0]][0]; TRIVIAL_F1 = 2 * y0.sum() / (2 * y0.sum() + (~y0).sum())
n = len(y0); rng = np.random.default_rng(2026)
if HAS_CAL:
    METRICS["mcc_cal"] = mcc; SRC["mcc_cal"] = "A_cal"
D = {"A": A, "A_cal": A_cal}
W = {m: pd.DataFrame({c: {s: fn(*D[SRC[m]][s][c]) for s in SEEDS} for c in CONDS}) for m, fn in METRICS.items()}

def boot(metric, a, b):
    fn = METRICS[metric]; S_ = D[SRC[metric]]; out = np.empty(args.boot)
    for i in range(args.boot):
        d = []
        for s in rng.choice(SEEDS, len(SEEDS)):
            idx = rng.integers(0, n, n); y = S_[s][a][0][idx]
            d.append(fn(y, S_[s][a][1][idx]) - fn(y, S_[s][b][1][idx]))
        out[i] = np.nanmean(d)
    return out

comps = [("Q1", "synth_only", "real_full"), ("Q2a", "btl", "real_small"), ("Q2b", "btl", "synth_only"), ("ref", "btl", "real_full")]
rows = []
for metric in METRICS:
    for name, a, b in comps:
        if a not in CONDS or b not in CONDS: continue
        d = (W[metric][a] - W[metric][b]).values; bd = boot(metric, a, b)
        try: pw = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
        except ValueError: pw = np.nan
        mc = []
        S_ = D[SRC[metric]]
        for s in SEEDS:
            y = S_[s][a][0]; ca = (S_[s][a][1] >= THR) == y; cb = (S_[s][b][1] >= THR) == y
            n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
            mc.append(stats.binomtest(n01, n01 + n10).pvalue if n01 + n10 else 1.0)
        rows.append(dict(metric=metric, comparison=name, a=a, b=b, mean_a=W[metric][a].mean(), mean_b=W[metric][b].mean(),
                         mean_diff=d.mean(), ci95_low=np.nanpercentile(bd, 2.5), ci95_high=np.nanpercentile(bd, 97.5),
                         p_bootstrap=min(1, 2 * min((bd <= 0).mean(), (bd >= 0).mean())),
                         p_ttest=stats.ttest_rel(W[metric][a], W[metric][b]).pvalue, p_wilcoxon=pw,
                         seeds_a_better=int((d > 0).sum()), n_seeds=len(d),
                         mcnemar_seeds_sig=int(sum(p < .05 for p in mc)) if metric in ("mcc", "mcc_cal") else np.nan,
                         per_seed_diff=";".join(f"{x:+.4f}" for x in d)))
C = pd.DataFrame(rows)
for metric in METRICS:                      # Holm across the three primary comparisons, per metric
    m = (C.metric == metric) & C.comparison.isin(["Q1", "Q2a", "Q2b"]); pv = C.loc[m, "p_bootstrap"].values
    order = np.argsort(pv); adj = np.empty(len(pv)); run = 0
    for r, i in enumerate(order): run = max(run, min(1, (len(pv) - r) * pv[i])); adj[i] = run
    C.loc[m, "p_holm"] = adj
C.round(5).to_csv(OUT / "comparisons_v2.csv", index=False)

# per-condition summary + triviality check
summ = pd.DataFrame({m: W[m].mean() for m in METRICS}).T.round(4)
triv = pd.DataFrame({c: {"seeds_F1_above_trivial": int((W["f1"][c] > TRIVIAL_F1 + 1e-4).sum()),
                         "seeds_MCC_above_0.1": int((W["mcc"][c] > 0.1).sum())} for c in CONDS})
pd.concat([summ, triv]).to_csv(OUT / "summary_v2.csv")

# gap closure on ROC-AUC
ref = C[(C.metric == "roc_auc") & (C.comparison == "ref")].iloc[0]
tier = ("CLOSED" if ref.ci95_low >= -MARGIN else "LIKELY CLOSED" if ref.mean_diff >= -MARGIN
        else "PARTIALLY CLOSED" if ref.mean_diff >= -PARTIAL else "NOT CLOSED")
rec = {}
for m in ("mcc", "roc_auc"):
    for base in ("real_small", "synth_only"):
        gap = W[m].real_full - W[m][base]
        rec[f"{m}_recovered_vs_{base}"] = float(np.nanmean(np.where(gap > 1e-3, (W[m].btl - W[m][base]) / gap, np.nan)))
verdict = dict(metric="roc_auc", mean_btl_minus_full=ref.mean_diff, ci95_low=ref.ci95_low, ci95_high=ref.ci95_high,
               tier=tier, trivial_f1=TRIVIAL_F1, **rec)
pd.Series(verdict).to_csv(OUT / "gap_verdict_v2.csv", header=["value"])
for m in METRICS: W[m].round(4).to_csv(OUT / f"per_seed_{m}.csv")

try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    lab = {"real_full": "Real-only\n(full pool)", "real_small": "Real-only\n(11 images)", "synth_only": "Synthetic-only", "btl": "BTL"}
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), dpi=200)
    for ax, (m, title) in zip(axes, [("mcc", "MCC (primary)"), ("roc_auc", "ROC-AUC"), ("f1", "F1")]):
        for i, c in enumerate(CONDS):
            v = W[m][c].values; ax.bar(i, np.nanmean(v), color="#BFBFBF", edgecolor="#333", width=0.6)
            ax.scatter(np.full(len(v), i) + np.linspace(-.12, .12, len(v)), v, s=10, color="#222", zorder=3)
        if m == "f1": ax.axhline(TRIVIAL_F1, ls="--", color="#555", lw=1); ax.text(3.4, TRIVIAL_F1, "always-\npneumonia", fontsize=6, va="center")
        ax.set_xticks(range(len(CONDS)), [lab[c] for c in CONDS], fontsize=7); ax.set_title(title, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "metrics_by_condition_v2.png")
except Exception as e:
    print("figure skipped:", e)

pd.set_option("display.width", 220)
print("Mean across seeds:\n" + pd.concat([summ, triv]).to_string())
print(f"\nAlways-pneumonia F1 = {TRIVIAL_F1:.4f}")
print("\nComparisons:\n" + C[["metric", "comparison", "a", "b", "mean_diff", "ci95_low", "ci95_high", "p_bootstrap", "p_holm",
                               "p_ttest", "p_wilcoxon", "seeds_a_better"]].round(4).to_string(index=False))
print("\nGAP VERDICT (ROC-AUC):", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in verdict.items()})
