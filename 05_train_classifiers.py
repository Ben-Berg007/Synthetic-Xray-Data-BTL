"""
Step 5: Train and test the four classifier conditions for each seed.

Conditions (all hyperparameters fixed a priori; the only difference between REAL_SMALL and BTL is
the starting weights, so any gap between them is the effect of synthetic pretraining):
  real_full   ImageNet ResNet-18 -> all dev_train real images (3,896); early stopping on the real
              validation set (F1), max 10 epochs, patience 3.            [Q1 benchmark]
  real_small  ImageNet ResNet-18 -> the seed's 11 real images, 150 steps.  [fair Q2 baseline]
  synth_only  ImageNet ResNet-18 -> the seed's 2,000 synthetic images, 5 epochs, no real data.
  btl         synth_only weights -> the same 11 real images, 150 steps (identical to real_small).
Every model is scored ONCE on the untouched held-out test set (1,222 images) at threshold 0.5.
Loss: BCE with pos_weight = n_normal / n_pneumonia of the training set (handles imbalance).

Run from D:\\School\\synthetic_data:
    python scripts\\05_train_classifiers.py                 (all seeds, ~30-60 min on the 3060)
    python scripts\\05_train_classifiers.py --seeds 0       (one seed)
Outputs (outputs\\step5\\): predictions.csv (one row per test image x condition x seed),
    metrics.csv, train_log.csv, cache_*.npz (preprocessed image caches, safe to delete)
"""
import argparse, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from sklearn.metrics import (f1_score, precision_score, recall_score, accuracy_score,
                             average_precision_score, roc_auc_score, confusion_matrix)
from torchvision import models
from torchvision.transforms import v2 as T

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
ap.add_argument("--conditions", nargs="+", default=["real_full", "real_small", "synth_only", "btl"])
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--no_pretrained", action="store_true", help="smoke tests only")
ap.add_argument("--smoke", action="store_true", help="tiny run to check the pipeline")
args = ap.parse_args()
dev = args.device
OUT = ROOT / "outputs" / "step5"; OUT.mkdir(parents=True, exist_ok=True)
CACHE_RES, IMG = 256, 224
HP = dict(bs=32, lr=1e-4, wd=1e-4, full_max_epochs=10, full_patience=3, synth_epochs=5,
          small_steps=150, threshold=0.5)
if args.smoke:
    HP.update(full_max_epochs=1, synth_epochs=1, small_steps=3)

# ---------------- image caches (padded to square, 256x256 grayscale uint8) ----------------
def load_img(p):
    return np.asarray(ImageOps.pad(Image.open(p).convert("L"), (CACHE_RES, CACHE_RES), color=0), dtype=np.uint8)

def cached(name, paths):
    f = OUT / f"cache_{name}.npz"
    if f.exists():
        z = np.load(f, allow_pickle=True)
        if list(z["paths"]) == list(paths):
            return z["x"]
    t0 = time.time(); print(f"building cache {name} ({len(paths)} images)...")
    x = np.stack([load_img(p) for p in paths])
    np.savez(f, x=x, paths=np.array(paths, dtype=object)); print(f"  done in {time.time()-t0:.0f}s")
    return x

asg = pd.read_csv(ROOT / "outputs" / "step2" / "assignments.csv")
man = pd.read_csv(ROOT / "outputs" / "step1" / "manifest.csv")
fs_all = pd.read_csv(ROOT / "outputs" / "step2" / "fewshot_sets.csv")
test = man[man.split == "test_holdout"].reset_index(drop=True)
trn = asg[asg.role == "dev_train"].reset_index(drop=True)
val = asg[asg.role == "val"].reset_index(drop=True)
if args.smoke:
    test, trn, val = test.head(40), trn.head(60), val.head(20)
Xte = cached("test" + ("_smoke" if args.smoke else ""), [str(ROOT / p) for p in test.path]); yte = test.y.values
Xtr = cached("dev_train" + ("_smoke" if args.smoke else ""), [str(ROOT / p) for p in trn.path]); ytr = trn.y.values
Xva = cached("val" + ("_smoke" if args.smoke else ""), [str(ROOT / p) for p in val.path]); yva = val.y.values

# ---------------- model / data helpers ----------------
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
aug = T.Compose([T.RandomResizedCrop(IMG, scale=(0.8, 1.0), ratio=(0.9, 1.1), antialias=True),
                 T.RandomRotation(7), T.ColorJitter(brightness=0.1, contrast=0.1)])
ev = T.Resize((IMG, IMG), antialias=True)

def to_input(xb_uint8, train):
    x = torch.from_numpy(xb_uint8).unsqueeze(1).to(dev).float() / 255.      # B,1,H,W
    x = torch.stack([aug(i) for i in x]) if train else ev(x)
    return (x.repeat(1, 3, 1, 1) - MEAN) / STD

def new_model():
    w = None if args.no_pretrained else models.ResNet18_Weights.IMAGENET1K_V1
    m = models.resnet18(weights=w); m.fc = nn.Linear(m.fc.in_features, 1)
    return m.to(dev)

def loss_fn(y):
    pw = torch.tensor([(y == 0).sum() / max((y == 1).sum(), 1)], device=dev, dtype=torch.float)
    return nn.BCEWithLogitsLoss(pos_weight=pw)

@torch.no_grad()
def predict(m, X, bs=128):
    m.eval(); out = []
    for i in range(0, len(X), bs):
        with torch.autocast(dev, enabled=dev == "cuda", dtype=torch.float16):
            out.append(torch.sigmoid(m(to_input(X[i:i + bs], False)).float()).squeeze(1).cpu())
    return torch.cat(out).numpy()

def train_steps(m, X, y, steps, log, tag):
    """fixed number of optimisation steps; batch = whole set if smaller than bs"""
    opt = torch.optim.AdamW(m.parameters(), lr=HP["lr"], weight_decay=HP["wd"]); lf = loss_fn(y)
    scaler = torch.amp.GradScaler(dev, enabled=dev == "cuda"); rng = np.random.default_rng(0)
    bs = min(HP["bs"], len(X)); m.train()
    for s in range(steps):
        idx = rng.choice(len(X), size=bs, replace=False)
        with torch.autocast(dev, enabled=dev == "cuda", dtype=torch.float16):
            loss = lf(m(to_input(X[idx], True)).squeeze(1).float(), torch.tensor(y[idx], device=dev).float())
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        if (s + 1) % 25 == 0 or s + 1 == steps:
            log.append(dict(tag=tag, step=s + 1, loss=loss.item()))
    return m

def train_epochs(m, X, y, epochs, log, tag, Xv=None, yv=None, patience=None):
    opt = torch.optim.AdamW(m.parameters(), lr=HP["lr"], weight_decay=HP["wd"]); lf = loss_fn(y)
    scaler = torch.amp.GradScaler(dev, enabled=dev == "cuda"); rng = np.random.default_rng(0)
    best, best_state, bad = -1, None, 0
    for ep in range(epochs):
        m.train(); perm = rng.permutation(len(X)); tot = 0
        for i in range(0, len(X), HP["bs"]):
            idx = perm[i:i + HP["bs"]]
            with torch.autocast(dev, enabled=dev == "cuda", dtype=torch.float16):
                loss = lf(m(to_input(X[idx], True)).squeeze(1).float(), torch.tensor(y[idx], device=dev).float())
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); tot += loss.item() * len(idx)
        rec = dict(tag=tag, epoch=ep + 1, loss=tot / len(X))
        if Xv is not None:
            f1 = f1_score(yv, predict(m, Xv) >= HP["threshold"], zero_division=0); rec["val_f1"] = f1
            if f1 > best:
                best, bad = f1, 0; best_state = {k: v.detach().clone() for k, v in m.state_dict().items()}
            else:
                bad += 1
        log.append(rec); print(f"    {tag} epoch {ep+1}: {rec}")
        if patience and bad >= patience:
            break
    if best_state is not None:
        m.load_state_dict(best_state)
    return m

def metrics(y, p, thr):
    yh = (p >= thr).astype(int); tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    return dict(f1=f1_score(y, yh, zero_division=0), precision=precision_score(y, yh, zero_division=0),
                recall=recall_score(y, yh, zero_division=0), specificity=tn / max(tn + fp, 1),
                accuracy=accuracy_score(y, yh), pr_auc=average_precision_score(y, p),
                roc_auc=roc_auc_score(y, p) if len(set(y)) > 1 else np.nan, tn=tn, fp=fp, fn=fn, tp=tp)

# ---------------- run ----------------
preds, mets, log = [], [], []
for seed in args.seeds:
    print(f"\n===== seed {seed} =====")
    fs = fs_all[fs_all.seed == seed]
    Xs = cached(f"fewshot_seed{seed}", [str(ROOT / p) for p in fs.path]); ys = fs.y.values
    sdir = ROOT / "outputs" / "step4" / "synthetic" / f"seed_{seed}"
    sm = pd.read_csv(sdir / "synthetic_manifest.csv"); sm = sm[sm.kept == 1]
    if args.smoke: sm = sm.groupby("label").head(10)
    Xsy = cached(f"synth_seed{seed}" + ("_smoke" if args.smoke else ""), [str(sdir / f) for f in sm.file]); ysy = sm.y.values
    trained = {}
    for cond in ["real_full", "real_small", "synth_only", "btl"]:
        if cond not in args.conditions and not (cond == "synth_only" and "btl" in args.conditions):
            continue
        torch.manual_seed(1000 + seed); np.random.seed(1000 + seed); t0 = time.time()
        tag = f"seed{seed}_{cond}"; print(f"  training {cond}...")
        if cond == "real_full":
            m = train_epochs(new_model(), Xtr, ytr, HP["full_max_epochs"], log, tag, Xva, yva, HP["full_patience"])
        elif cond == "real_small":
            m = train_steps(new_model(), Xs, ys, HP["small_steps"], log, tag)
        elif cond == "synth_only":
            m = train_epochs(new_model(), Xsy, ysy, HP["synth_epochs"], log, tag)
        else:  # btl: start from this seed's synth_only weights, then identical recipe to real_small
            m = new_model(); m.load_state_dict(trained["synth_only"])
            m = train_steps(m, Xs, ys, HP["small_steps"], log, tag)
        trained[cond] = {k: v.detach().clone() for k, v in m.state_dict().items()}
        if cond not in args.conditions:
            continue
        p = predict(m, Xte)
        preds.append(pd.DataFrame(dict(seed=seed, condition=cond, test_idx=np.arange(len(p)),
                                       patient_id=test.patient_id.values, y=yte, prob=np.round(p, 5))))
        mets.append(dict(seed=seed, condition=cond, minutes=round((time.time() - t0) / 60, 2),
                         **{k: (round(float(v), 4) if isinstance(v, float) else int(v)) for k, v in metrics(yte, p, HP["threshold"]).items()}))
        print(f"  {cond}: " + ", ".join(f"{k}={mets[-1][k]}" for k in ["f1", "precision", "recall", "specificity", "pr_auc", "fn"]))
    # save after every seed so a crash loses little
    for name, df in (("predictions", pd.concat(preds)), ("metrics", pd.DataFrame(mets)), ("train_log", pd.DataFrame(log))):
        f = OUT / f"{name}{'_smoke' if args.smoke else ''}.csv"
        if name != "train_log" and f.exists():          # keep results from seeds/conditions not re-run
            old = pd.read_csv(f)
            old = old[~(old.seed.isin(args.seeds) & old.condition.isin(args.conditions))]
            df = pd.concat([old, df])
        df.to_csv(f, index=False)
pd.Series(HP).to_csv(OUT / "hyperparameters.csv", header=["value"])
print("\n" + pd.DataFrame(mets).to_string(index=False))
