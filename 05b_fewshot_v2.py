"""
Step 5b (RUN 2): revised few-shot training recipe, selected on the VALIDATION set only.
Protocol pre-registered 2026-09-30 (see capstone-study-design.md, "Run 2 protocol").

Why: run 1 trained the whole ResNet-18 for 150 steps on 11 images with no regularisation; the
models memorised the images (train loss ~2e-4) and predicted "pneumonia" for almost every test image.

Fixed for every run-2 few-shot recipe: dropout 0.5 before the final layer, label smoothing 0.1,
weight decay 1e-2, stronger augmentation, frozen layers keep batch-norm in eval mode, threshold 0.5.
Candidate grid (9): trainable part {head, layer4+head, all} x steps {25, 50, 100}.
Synthetic-only run 2: same head, early stopping on 10% held-out SYNTHETIC images (no real data).

STEP A - choose the recipe (validation set only, test set untouched). Three variants (amendment 2026-09-30):
    python scripts\\05b_fewshot_v2.py --mode select                               (done: no CLAHE)
    python scripts\\05b_fewshot_v2.py --mode select --prep clahe
    python scripts\\05b_fewshot_v2.py --mode select --prep clahe --intensity_aug
    python scripts\\05b_fewshot_v2.py --mode compare        -> final_choice.json (highest mean val MCC)
STEP B - score once on the test set with the final choice (real_full retrained with the same preprocessing):
    python scripts\\05b_fewshot_v2.py --mode final
    -> outputs\\step5b\\predictions_v2.csv, metrics_v2.csv  (incl. secondary validation-calibrated threshold)
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from sklearn.metrics import (f1_score, matthews_corrcoef, balanced_accuracy_score, precision_score,
                             recall_score, roc_auc_score, average_precision_score, confusion_matrix)
from torchvision import models
from torchvision.transforms import v2 as T

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["select", "compare", "final"], required=True)
ap.add_argument("--prep", choices=["none", "clahe"], default="none")
ap.add_argument("--intensity_aug", action="store_true")
ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--no_pretrained", action="store_true", help="smoke tests only")
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()
dev = args.device
S5, OUT = ROOT / "outputs" / "step5", ROOT / "outputs" / "step5b"
OUT.mkdir(parents=True, exist_ok=True)
SFX = "_smoke" if args.smoke else ""
if args.mode == "final":                       # final mode takes the variant from final_choice.json
    _fc = json.load(open(OUT / f"final_choice{SFX}.json")); args.prep, args.intensity_aug = _fc["prep"], _fc["intensity_aug"]
VAR = ("" if args.prep == "none" else "_clahe") + ("_iaug" if args.intensity_aug else "")
CLAHE = dict(clip=3.0, tile=8)
IMG, THR, LS, WD, DROP = 224, 0.5, 0.1, 1e-2, 0.5
GRID = [dict(train=t, lr=lr, steps=s) for t, lr in (("head", 1e-3), ("layer4", 3e-4), ("all", 1e-4))
        for s in (25, 50, 100)]
SYN = dict(bs=32, lr=1e-4, max_epochs=10, patience=2, holdout=0.10)
if args.smoke:
    GRID = [dict(g, steps=2) for g in GRID]; SYN.update(max_epochs=2)

# ---------------- data (reuses run-1 caches; builds any that are missing) ----------------
def load_img(p):
    return np.asarray(ImageOps.pad(Image.open(p).convert("L"), (256, 256), color=0), dtype=np.uint8)
def _cached_raw(name, paths):
    f = S5 / f"cache_{name}.npz"
    if f.exists():
        z = np.load(f, allow_pickle=True)
        if list(z["paths"]) == list(paths):
            return z["x"]
    print(f"building cache {name} ({len(paths)} images)")
    x = np.stack([load_img(p) for p in paths]); np.savez(f, x=x, paths=np.array(paths, dtype=object)); return x
def cached(name, paths):
    x = _cached_raw(name, paths)
    if args.prep == "clahe":                      # label-blind, identical for every image (real and synthetic)
        import cv2
        c = cv2.createCLAHE(clipLimit=CLAHE["clip"], tileGridSize=(CLAHE["tile"], CLAHE["tile"]))
        x = np.stack([c.apply(i) for i in x])
    return x

man = pd.read_csv(ROOT / "outputs" / "step1" / "manifest.csv")
asg = pd.read_csv(ROOT / "outputs" / "step2" / "assignments.csv")
fs_all = pd.read_csv(ROOT / "outputs" / "step2" / "fewshot_sets.csv")
test = man[man.split == "test_holdout"].reset_index(drop=True)
val = asg[asg.role == "val"].reset_index(drop=True)
if args.smoke:
    test, val = test.head(40), val.head(20)
Xva, yva = cached("val" + SFX, [str(ROOT / p) for p in val.path]), val.y.values
if args.mode == "final":
    Xte, yte = cached("test" + SFX, [str(ROOT / p) for p in test.path]), test.y.values

# ---------------- model helpers ----------------
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
aug = T.Compose([T.RandomResizedCrop(IMG, scale=(0.7, 1.0), ratio=(0.9, 1.1), antialias=True),
                 T.RandomRotation(10), T.ColorJitter(brightness=0.2, contrast=0.2)])
ev = T.Resize((IMG, IMG), antialias=True)
AMP = dev == "cuda"

def to_input(xb, train):
    x = torch.from_numpy(xb).unsqueeze(1).to(dev).float() / 255.
    x = torch.stack([aug(i) for i in x]) if train else ev(x)
    if train and args.intensity_aug:              # brightness/contrast +-30%, gamma 0.7-1.4 (label-blind)
        b = x.shape[0]; r = lambda lo, hi: torch.empty(b, 1, 1, 1, device=dev).uniform_(lo, hi)
        m = x.mean(dim=(2, 3), keepdim=True)
        x = ((x - m) * r(0.7, 1.3) + m * r(0.7, 1.3)).clamp(0, 1) ** r(0.7, 1.4)
    return (x.repeat(1, 3, 1, 1) - MEAN) / STD

def new_model(plain=False):
    m = models.resnet18(weights=None if args.no_pretrained else models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(512, 1) if plain else nn.Sequential(nn.Dropout(DROP), nn.Linear(512, 1))
    return m.to(dev)

def fit_real_full(X, y, Xv, yv):
    """run-1 benchmark recipe (unchanged): whole network, plain head, lr 1e-4, wd 1e-4, max 10 epochs,
    early stopping on validation F1 (patience 3), standard BCE with pos_weight"""
    m = new_model(plain=True); set_trainable(m, "all")
    pw = torch.tensor([(y == 0).sum() / max((y == 1).sum(), 1)], device=dev, dtype=torch.float); lf = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=1e-4); scaler = torch.amp.GradScaler(dev, enabled=AMP)
    rng = np.random.default_rng(0); best, state, bad = -1, None, 0
    for ep in range(2 if args.smoke else 10):
        m.train(); perm = rng.permutation(len(X))
        for i in range(0, len(X), 32):
            b = perm[i:i + 32]
            with torch.autocast(dev, enabled=AMP, dtype=torch.float16):
                loss = lf(m(to_input(X[b], True)).squeeze(1).float(), torch.tensor(y[b], device=dev).float())
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        f = f1_score(yv, predict(m, Xv) >= THR, zero_division=0); print(f"    real_full epoch {ep+1}: val F1 {f:.4f}")
        if f > best: best, bad, state = f, 0, {k: v.detach().clone() for k, v in m.state_dict().items()}
        else: bad += 1
        if bad >= 3: break
    m.load_state_dict(state); return m

def set_trainable(m, part):
    for n, p in m.named_parameters():
        p.requires_grad = part == "all" or n.startswith("fc") or (part == "layer4" and n.startswith("layer4"))

def train_mode(m):
    m.train()
    for mod in m.modules():                      # frozen batch-norm layers keep their pretrained statistics
        if isinstance(mod, nn.BatchNorm2d) and not any(p.requires_grad for p in mod.parameters()):
            mod.eval()

def loss_fn(y):
    pw = torch.tensor([(y == 0).sum() / max((y == 1).sum(), 1)], device=dev, dtype=torch.float)
    bce = nn.BCEWithLogitsLoss(pos_weight=pw)
    return lambda logit, t: bce(logit, t * (1 - LS) + 0.5 * LS)

@torch.no_grad()
def predict(m, X, bs=128):
    m.eval(); out = []
    for i in range(0, len(X), bs):
        with torch.autocast(dev, enabled=AMP, dtype=torch.float16):
            out.append(torch.sigmoid(m(to_input(X[i:i + bs], False)).float()).squeeze(1).cpu())
    return torch.cat(out).numpy()

def fit_fewshot(m, X, y, rec):
    set_trainable(m, rec["train"]); lf = loss_fn(y)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=rec["lr"], weight_decay=WD)
    scaler = torch.amp.GradScaler(dev, enabled=AMP); train_mode(m)
    for _ in range(rec["steps"]):
        with torch.autocast(dev, enabled=AMP, dtype=torch.float16):
            loss = lf(m(to_input(X, True)).squeeze(1).float(), torch.tensor(y, device=dev).float())
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    return m, loss.item()

def fit_synth(X, y, seed):
    rng = np.random.default_rng(seed); idx = rng.permutation(len(X)); nh = int(len(X) * SYN["holdout"])
    hi, ti = idx[:nh], idx[nh:]
    m = new_model(); set_trainable(m, "all"); lf = loss_fn(y[ti])
    opt = torch.optim.AdamW(m.parameters(), lr=SYN["lr"], weight_decay=WD); scaler = torch.amp.GradScaler(dev, enabled=AMP)
    best, state, bad, log = 1e9, None, 0, []
    for ep in range(SYN["max_epochs"]):
        train_mode(m); perm = rng.permutation(ti)
        for i in range(0, len(perm), SYN["bs"]):
            b = perm[i:i + SYN["bs"]]
            with torch.autocast(dev, enabled=AMP, dtype=torch.float16):
                loss = lf(m(to_input(X[b], True)).squeeze(1).float(), torch.tensor(y[b], device=dev).float())
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        ph = torch.tensor(predict(m, X[hi])).clamp(1e-6, 1 - 1e-6)
        hl = nn.functional.binary_cross_entropy(ph, torch.tensor(y[hi]).float()).item(); log.append(round(hl, 4))
        if hl < best: best, bad, state = hl, 0, {k: v.detach().clone() for k, v in m.state_dict().items()}
        else: bad += 1
        if bad >= SYN["patience"]: break
    m.load_state_dict(state); return m, log

def metrics(y, p):
    yh = (p >= THR).astype(int); tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    return dict(mcc=matthews_corrcoef(y, yh), f1=f1_score(y, yh, zero_division=0), bal_acc=balanced_accuracy_score(y, yh),
                precision=precision_score(y, yh, zero_division=0), recall=recall_score(y, yh, zero_division=0),
                specificity=tn / max(tn + fp, 1), roc_auc=roc_auc_score(y, p) if len(set(y)) > 1 else np.nan,
                pr_auc=average_precision_score(y, p), tn=tn, fp=fp, fn=fn, tp=tp)

def seed_data(seed):
    fs = fs_all[fs_all.seed == seed]
    Xs = cached(f"fewshot_seed{seed}", [str(ROOT / p) for p in fs.path])
    sdir = ROOT / "outputs" / "step4" / "synthetic" / f"seed_{seed}"
    sm = pd.read_csv(sdir / "synthetic_manifest.csv"); sm = sm[sm.kept == 1]
    if args.smoke: sm = sm.groupby("label").head(10)
    Xsy = cached(f"synth_seed{seed}{SFX}", [str(sdir / f) for f in sm.file])
    return Xs, fs.y.values, Xsy, sm.y.values

def synth_model(seed, Xsy, ysy):
    f = OUT / f"synth_v2{VAR}_seed{seed}{SFX}.pt"
    if f.exists():
        m = new_model(); m.load_state_dict(torch.load(f, map_location=dev)); return m
    torch.manual_seed(1000 + seed); m, log = fit_synth(Xsy, ysy, seed)
    torch.save(m.state_dict(), f); print(f"  synth_v2 seed {seed}: holdout loss by epoch {log}"); return m

# ---------------- STEP A: select on validation ----------------
if args.mode == "select":
    rows, t0 = [], time.time()
    for seed in args.seeds:
        Xs, ys, Xsy, ysy = seed_data(seed); sm_state = synth_model(seed, Xsy, ysy).state_dict()
        for ri, rec in enumerate(GRID):
            for cond in ("real_small", "btl"):
                torch.manual_seed(1000 + seed); m = new_model()
                if cond == "btl": m.load_state_dict(sm_state)
                m, last = fit_fewshot(m, Xs, ys, rec)
                rows.append(dict(seed=seed, recipe=ri, **rec, condition=cond, train_loss=round(last, 4),
                                 **{k: round(float(v), 4) for k, v in metrics(yva, predict(m, Xva)).items()}))
        print(f"seed {seed} done ({(time.time()-t0)/60:.1f} min)")
    G = pd.DataFrame(rows); G.to_csv(OUT / f"val_grid{VAR}{SFX}.csv", index=False)
    cols = ["mcc", "f1", "bal_acc", "specificity", "roc_auc"]
    summ = G.groupby(["recipe", "train", "lr", "steps", "condition"])[cols].mean().unstack("condition")
    summ.columns = [f"{c}_{cond}" for c, cond in summ.columns]
    summ = summ.reset_index()
    summ["mean_val_mcc"] = summ.recipe.map(G.groupby("recipe").mcc.mean())
    table = summ.sort_values("mean_val_mcc", ascending=False).set_index("recipe")
    table.to_csv(OUT / f"val_summary{VAR}{SFX}.csv")
    best = G.groupby(["recipe", "steps"]).mcc.mean().reset_index().sort_values(["mcc", "steps"], ascending=[False, True]).iloc[0]
    chosen = dict(GRID[int(best.recipe)], recipe=int(best.recipe), mean_val_mcc=round(float(best.mcc), 4),
                  prep=args.prep, intensity_aug=args.intensity_aug,
                  selected_on="validation set only", seeds=args.seeds, frozen_at=time.strftime("%Y-%m-%d %H:%M"))
    json.dump(chosen, open(OUT / f"chosen_recipe{VAR}{SFX}.json", "w"), indent=2)
    pd.set_option("display.width", 220)
    print("\nValidation results (mean over seeds):\n" + table[["train","steps","mcc_real_small","mcc_btl","mean_val_mcc","specificity_real_small","specificity_btl","roc_auc_real_small","roc_auc_btl"]].round(3).to_string())
    print("\nCHOSEN RECIPE:", chosen)

# ---------------- compare variants (validation only) ----------------
elif args.mode == "compare":
    rows = []
    for f in sorted(OUT.glob(f"chosen_recipe*{SFX}.json")):
        if SFX == "" and "_smoke" in f.name: continue
        c = json.load(open(f)); c.setdefault("prep", "none"); c.setdefault("intensity_aug", False); c["file"] = f.name; rows.append(c)
    T_ = pd.DataFrame(rows).sort_values("mean_val_mcc", ascending=False)
    print(T_[["file", "prep", "intensity_aug", "train", "steps", "mean_val_mcc"]].to_string(index=False))
    win = T_.iloc[0].to_dict(); win = {k: (v.item() if hasattr(v, "item") else v) for k, v in win.items()}
    win["frozen_at"] = time.strftime("%Y-%m-%d %H:%M"); json.dump(win, open(OUT / f"final_choice{SFX}.json", "w"), indent=2, default=str)
    print("\nFINAL CHOICE:", win)

# ---------------- STEP B: final test scoring ----------------
else:
    rec = json.load(open(OUT / f"final_choice{SFX}.json")); print("Using frozen final choice:", rec)
    trn = asg[asg.role == "dev_train"].reset_index(drop=True)
    if args.smoke: trn = trn.head(60)
    Xtr = cached("dev_train" + SFX, [str(ROOT / p) for p in trn.path]); ytr = trn.y.values
    THRS = np.round(np.arange(0.05, 0.96, 0.01), 2)
    def cal_thr(pv):                               # secondary analysis: threshold maximising validation MCC
        sc = [matthews_corrcoef(yva, (pv >= t).astype(int)) for t in THRS]; return float(THRS[int(np.argmax(sc))])
    preds, mets = [], []
    for seed in args.seeds:
        Xs, ys, Xsy, ysy = seed_data(seed); sm = synth_model(seed, Xsy, ysy); sm_state = sm.state_dict()
        out = {"synth_only": sm}
        for cond in ("real_small", "btl"):
            torch.manual_seed(1000 + seed); m = new_model()
            if cond == "btl": m.load_state_dict(sm_state)
            out[cond] = fit_fewshot(m, Xs, ys, rec)[0]
        torch.manual_seed(1000 + seed); print(f"  seed {seed}: training real_full benchmark"); out["real_full"] = fit_real_full(Xtr, ytr, Xva, yva)
        for cond, m in out.items():
            p, pv = predict(m, Xte), predict(m, Xva); t = cal_thr(pv)
            preds.append(pd.DataFrame(dict(seed=seed, condition=cond, test_idx=np.arange(len(p)), patient_id=test.patient_id.values,
                                           y=yte, prob=np.round(p, 5), thr_val=t)))
            mm = {k: round(float(v), 4) for k, v in metrics(yte, p).items()}
            mc = {f"cal_{k}": round(float(v), 4) for k, v in metrics(yte, (p >= t).astype(float) * 0.999 + 0.0005).items()
                  if k in ("mcc", "f1", "bal_acc", "specificity", "recall")}
            mets.append(dict(seed=seed, condition=cond, thr_val=t, **mm, **mc))
            print(f"  seed {seed} {cond}: mcc={mm['mcc']} f1={mm['f1']} spec={mm['specificity']} auc={mm['roc_auc']} | calibrated thr {t}: mcc={mc['cal_mcc']}")
    P = pd.concat(preds); P.to_csv(OUT / f"predictions_v2{SFX}.csv", index=False)
    M = pd.DataFrame(mets); M.to_csv(OUT / f"metrics_v2{SFX}.csv", index=False)
    print("\n" + M.groupby("condition")[["mcc", "f1", "bal_acc", "specificity", "recall", "roc_auc", "cal_mcc", "cal_specificity", "thr_val"]].mean().round(3).to_string())
