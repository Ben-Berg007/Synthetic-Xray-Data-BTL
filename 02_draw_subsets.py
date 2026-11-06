"""
Step 2: Draw the validation set and the few-shot (10-11 image) sets for each seed.

Design (decided 2026-09-26):
  - Everything is drawn from the dev_pool only; test_holdout is never touched.
  - A patient-level validation set (~15% of dev patients) is carved out first. It is used
    only for early stopping / model selection of the full-pool real-only baseline.
  - For each of 10 seeds (0-9), 11 real images (5 NORMAL + 6 PNEUMONIA), one image per patient,
    are drawn from the remaining dev_train patients. That same set is used to
      (a) LoRA fine-tune the diffusion generator for that seed, and
      (b) fine-tune the BTL classifier and train the real-only (10-11 image) baseline,
    so every few-shot condition sees exactly the same real images.

Run from D:\\School\\synthetic_data:
    python scripts\\02_draw_subsets.py
Outputs (outputs\\step2\\):
    assignments.csv      manifest + role column (val / dev_train) for every dev image
    fewshot_sets.csv     one row per (seed, image)
    fewshot\\seed_k\\     copies of each seed's 11 images, 512x512 grayscale PNG, plus metadata.jsonl captions
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "step2"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = list(range(10))   # seeds 0-4 first run; 5-9 added 2026-09-27 (draws for 0-4 are unchanged)
N_PER_CLASS = {"NORMAL": 5, "PNEUMONIA": 6}          # 11 images per seed
VAL_FOLDS = 7                                          # 1 of 7 patient folds ~= 15% of dev patients
RES = 512
PROMPTS = {"NORMAL": "a chest x-ray, normal lungs",
           "PNEUMONIA": "a chest x-ray, pneumonia"}

man = pd.read_csv(ROOT / "outputs" / "step1" / "manifest.csv")
dev = man[man.split == "dev_pool"].reset_index(drop=True)

# 1) patient-level validation carve-out
sgkf = StratifiedGroupKFold(n_splits=VAL_FOLDS, shuffle=True, random_state=42)
_, val_idx = next(sgkf.split(dev, dev["y"], groups=dev["patient_id"]))
dev["role"] = "dev_train"
dev.loc[val_idx, "role"] = "val"
dev.to_csv(OUT / "assignments.csv", index=False)
assert not (set(dev[dev.role == "val"].patient_id) & set(dev[dev.role == "dev_train"].patient_id))

# 2) few-shot draws: one image per patient, stratified by class
train = dev[dev.role == "dev_train"]
rows = []
for seed in SEEDS:
    rng = np.random.default_rng(seed)
    for label, k in N_PER_CLASS.items():
        pool = train[train.label == label]
        patients = rng.choice(pool.patient_id.unique(), size=k, replace=False)
        for pid in patients:
            imgs = pool[pool.patient_id == pid]
            pick = imgs.iloc[rng.integers(len(imgs))]
            rows.append(dict(seed=seed, label=label, y=pick.y, subtype=pick.subtype,
                             patient_id=pid, path=pick.path, md5=pick.md5,
                             prompt=PROMPTS[label]))
fs = pd.DataFrame(rows)
fs.to_csv(OUT / "fewshot_sets.csv", index=False)

# 3) export each seed's images for LoRA training (grayscale, center-padded to square, 512x512)
for seed, g in fs.groupby("seed"):
    d = OUT / "fewshot" / f"seed_{seed}"
    d.mkdir(parents=True, exist_ok=True)
    meta = []
    for i, r in enumerate(g.itertuples()):
        im = Image.open(ROOT / r.path).convert("L")
        im = ImageOps.pad(im, (RES, RES), color=0, method=Image.BICUBIC)
        name = f"{r.label.lower()}_{i:02d}.png"
        im.save(d / name)
        meta.append({"file_name": name, "text": r.prompt, "label": r.label, "source_path": r.path})
    with open(d / "metadata.jsonl", "w") as f:
        for m in meta:
            f.write(json.dumps(m) + "\n")

# 4) summary
lines = [
    f"dev_pool images: {len(dev)}  | val: {(dev.role=='val').sum()} images / {dev[dev.role=='val'].patient_id.nunique()} patients"
    f"  | dev_train: {(dev.role=='dev_train').sum()} images / {train.patient_id.nunique()} patients",
    "val by label: " + dev[dev.role == "val"].label.value_counts().to_dict().__str__(),
    "",
    "Few-shot sets (images per seed by label):",
    fs.groupby(["seed", "label"]).size().unstack().to_string(),
    "",
    f"Unique images across all seeds: {fs.md5.nunique()} of {len(fs)}",
    f"Pneumonia subtypes drawn: {fs[fs.label=='PNEUMONIA'].subtype.value_counts().to_dict()}",
]
(OUT / "summary.txt").write_text("\n".join(lines))
print("\n".join(lines))
