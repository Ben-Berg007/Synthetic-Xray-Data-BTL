"""
Step 6c: patient-level (cluster) bootstrap re-analysis of the run 2 predictions.
Added 2026-10-29 in response to instructor feedback on repeated patient images.

Identical to 06b_stats_v2.py except the inner resampling unit: 06b resamples test IMAGES
within each seed; 06c resamples test PATIENTS (all of a patient's images together), because
the 1,222 hold-out images come from 624 patients and images from one patient are correlated.
Seeds are still resampled in the outer level. Same metrics, comparisons, Holm correction,
and gap-closure tiers as 06b. No models are retrained and no new predictions are made.

    python scripts\\06c_patient_bootstrap.py [--file outputs\\step5b\\predictions_v2.csv] [--boot 1000]
Outputs in outputs\\step6c\\.
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--file", default=str(ROOT / "outputs" / "step5b" / "predictions_v2.csv"))
ap.add_argument("--image_level", default=str(ROOT / "outputs" / "step6b" / "comparisons_v2.csv"))
ap.add_argument("--out", default=str(ROOT / "outputs" / "step6c"))
ap.add_argument("--boot", type=int, default=1000)
args = ap.parse_args()
OUT = Path(args.out); OUT.mkdir(parents=True, exist_ok=True)
P = pd.read_csv(args.file); THR, MARGIN, PARTIAL = 0.5, 0.05, 0.10
CONDS = ["real_full", "real_small", "synth_only", "btl"]
SEEDS = sorted(P.seed.unique())

def mcc(y, p):
    yh = p >= THR; tp = float((y & yh).sum()); tn = float((~y & ~yh).sum()); fp = float((~y & yh).sum()); fn = float((y & ~yh).sum())
    d = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)); return (tp * tn - fp * fn) / d if d else 0.0
def f1(y, p):
    yh = p >= THR; tp = (y & yh).sum(); return 2 * tp / max(2 * tp + (~y & yh).sum() + (y & ~yh).sum(), 1)
def auc(y, p):
    return roc_auc_score(y, p) if y.any() and (~y).any() else np.nan
def sens(y, p):
    yh = p >= THR; return (y & yh).sum() / max(y.sum(), 1)
METRICS = {"mcc": mcc, "f1": f1, "roc_auc": auc, "mcc_cal": mcc, "sensitivity": sens}
SRC = {"mcc": "A", "f1": "A", "roc_auc": "A", "mcc_cal": "A_cal", "sensitivity": "A"}

A, A_cal = {}, {}
for (s, c), g in P.groupby(["seed", "condition"]):
    g = g.sort_values("test_idx"); y = g.y.values.astype(bool); p = g.prob.values; A.setdefault(s, {})[c] = (y, p)
    t = float(g.thr_val.iloc[0]); pc = np.where(p >= t, 0.5 + 0.5 * (p - t) / max(1 - t, 1e-6), 0.5 * p / max(t, 1e-6))
    A_cal.setdefault(s, {})[c] = (y, pc)
D = {"A": A, "A_cal": A_cal}

# patient clusters: test_idx -> patient (identical order for every seed/condition)
ref = P[(P.seed == SEEDS[0]) & (P.condition == CONDS[0])].sort_values("test_idx")
pid = ref.patient_id.values
for (s, c), g in P.groupby(["seed", "condition"]):
    assert (g.sort_values("test_idx").patient_id.values == pid).all()
patients = pd.unique(pid)
groups = [np.flatnonzero(pid == q) for q in patients]
n_img, n_pat = len(pid), len(patients)

W = {m: pd.DataFrame({c: {s: fn(*D[SRC[m]][s][c]) for s in SEEDS} for c in CONDS}) for m, fn in METRICS.items()}
rng = np.random.default_rng(2026)

def boot(metric, a, b):
    fn = METRICS[metric]; S_ = D[SRC[metric]]; out = np.empty(args.boot)
    for i in range(args.boot):
        d = []
        for s in rng.choice(SEEDS, len(SEEDS)):
            idx = np.concatenate([groups[k] for k in rng.integers(0, n_pat, n_pat)])   # resample patients
            y = S_[s][a][0][idx]
            d.append(fn(y, S_[s][a][1][idx]) - fn(y, S_[s][b][1][idx]))
        out[i] = np.nanmean(d)
    return out

comps = [("Q1", "synth_only", "real_full"), ("Q2a", "btl", "real_small"), ("Q2b", "btl", "synth_only"), ("ref", "btl", "real_full"),
         ("guard", "btl", "real_full")]
rows = []
for metric in METRICS:
    for name, a, b in comps:
        if (metric == "sensitivity") != (name == "guard"): continue
        d = (W[metric][a] - W[metric][b]).values; bd = boot(metric, a, b)
        rows.append(dict(metric=metric, comparison=name, a=a, b=b, mean_diff=d.mean(),
                         ci95_low=np.nanpercentile(bd, 2.5), ci95_high=np.nanpercentile(bd, 97.5),
                         p_bootstrap=min(1, 2 * min((bd <= 0).mean(), (bd >= 0).mean()))))
C = pd.DataFrame(rows)
for metric in ["mcc", "f1", "roc_auc", "mcc_cal"]:   # Holm across Q1/Q2a/Q2b per metric, as in 06b
    m = (C.metric == metric) & C.comparison.isin(["Q1", "Q2a", "Q2b"]); pv = C.loc[m, "p_bootstrap"].values
    order = np.argsort(pv); adj = np.empty(len(pv)); run = 0
    for r, i in enumerate(order): run = max(run, min(1, (len(pv) - r) * pv[i])); adj[i] = run
    C.loc[m, "p_holm"] = adj

# side-by-side with the image-level results from 06b
try:
    I = pd.read_csv(args.image_level)[["metric", "comparison", "ci95_low", "ci95_high", "p_holm"]]
    I = I.rename(columns={"ci95_low": "img_ci95_low", "ci95_high": "img_ci95_high", "p_holm": "img_p_holm"})
    C = C.merge(I, on=["metric", "comparison"], how="left")
    C["ci_width_ratio_patient_vs_image"] = (C.ci95_high - C.ci95_low) / (C.img_ci95_high - C.img_ci95_low)
except FileNotFoundError:
    pass
C.round(5).to_csv(OUT / "comparisons_patient_level.csv", index=False)

r = C[(C.metric == "roc_auc") & (C.comparison == "ref")].iloc[0]
tier = ("CLOSED" if r.ci95_low >= -MARGIN else "LIKELY CLOSED" if r.mean_diff >= -MARGIN
        else "PARTIALLY CLOSED" if r.mean_diff >= -PARTIAL else "NOT CLOSED")
g = C[C.comparison == "guard"].iloc[0]
lines = [f"test images: {n_img} | test patients: {n_pat} | images per patient mean {n_img/n_pat:.2f}, max {max(len(x) for x in groups)}",
         f"patients with >1 image: {sum(len(x) > 1 for x in groups)}",
         f"bootstrap resamples: {args.boot} (seeds and patients resampled)",
         f"H3 (ROC-AUC, BTL - real_full): {r.mean_diff:+.3f} [{r.ci95_low:+.3f}, {r.ci95_high:+.3f}] -> {tier}",
         f"recall guard (sensitivity, BTL - real_full): {g.mean_diff:+.3f} [{g.ci95_low:+.3f}, {g.ci95_high:+.3f}]"]
(OUT / "summary_patient_level.txt").write_text("\n".join(lines))
print("\n".join(lines))
print(C.round(3).to_string(index=False))
