"""
Step 1: Profile the Kermany chest X-ray dataset and build a patient-level split.

Run from D:\\School\\synthetic_data:
    python scripts\\01_profile_and_split.py

Outputs (in outputs\\step1\\):
    manifest.csv        one row per unique image (label, subtype, patient_id, size, hash, split)
    profile_summary.txt counts by class/subtype/split, patients, image sizes, imbalance ratio
    leakage_check.txt   confirms zero patient overlap between test and dev pool
"""
import hashlib
import re
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "chest_xray"
OUT = ROOT / "outputs" / "step1"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
TEST_FRACTION_FOLDS = 5  # 1 of 5 patient-grouped folds -> ~20% held-out test

PNEU_RE = re.compile(r"^(person\d+)_(bacteria|virus)_\d+", re.I)
NORM_RE = re.compile(r"^((?:NORMAL2-)?IM-\d+)-\d+", re.I)


def parse(path: Path):
    name = path.stem
    label = path.parent.name.upper()
    m = PNEU_RE.match(name)
    if label == "PNEUMONIA" and m:
        return label, m.group(2).lower(), "P_" + m.group(1).lower()
    m = NORM_RE.match(name)
    if label == "NORMAL" and m:
        return label, "none", "N_" + m.group(1).upper()
    return label, "unknown", "UNPARSED_" + name  # treated as its own patient


rows, seen = [], set()
for p in sorted(DATA.rglob("*.jpeg")):
    if "__MACOSX" in p.parts or p.name.startswith("._"):
        continue
    if p.parent.name.upper() not in {"NORMAL", "PNEUMONIA"}:
        continue
    md5 = hashlib.md5(p.read_bytes()).hexdigest()
    if md5 in seen:  # the Kaggle zip contains a nested duplicate copy
        continue
    seen.add(md5)
    label, subtype, pid = parse(p)
    with Image.open(p) as im:
        w, h, mode = im.width, im.height, im.mode
    rows.append(dict(path=str(p.relative_to(ROOT)), kaggle_split=p.parent.parent.name,
                     label=label, y=int(label == "PNEUMONIA"), subtype=subtype,
                     patient_id=pid, width=w, height=h, mode=mode, md5=md5))

df = pd.DataFrame(rows)
if df.empty:
    raise SystemExit(f"No images found under {DATA}. Unzip the Kaggle download there first.")

# Patient-level, class-stratified held-out test split (created FIRST, before anything else)
sgkf = StratifiedGroupKFold(n_splits=TEST_FRACTION_FOLDS, shuffle=True, random_state=SEED)
_, test_idx = next(sgkf.split(df, df["y"], groups=df["patient_id"]))
df["split"] = "dev_pool"
df.loc[df.index[test_idx], "split"] = "test_holdout"
df.to_csv(OUT / "manifest.csv", index=False)

# Leakage check
overlap = set(df[df.split == "test_holdout"].patient_id) & set(df[df.split == "dev_pool"].patient_id)
(OUT / "leakage_check.txt").write_text(
    f"Patients in both test_holdout and dev_pool: {len(overlap)}\n" + "\n".join(sorted(overlap)))

# Profile
lines = [
    f"Unique images: {len(df)}  (duplicates removed by hash: {sum(1 for _ in DATA.rglob('*.jpeg')) - len(df)})",
    f"Unique patients: {df.patient_id.nunique()}  (unparsed filenames: {(df.subtype == 'unknown').sum()})",
    "", "Images by label:", df.label.value_counts().to_string(),
    f"Imbalance ratio (pneumonia:normal): {df.y.sum() / (len(df) - df.y.sum()):.2f} : 1",
    "", "Images by subtype:", df.subtype.value_counts().to_string(),
    "", "Patients by label:", df.groupby("label").patient_id.nunique().to_string(),
    "", "Images per patient:", df.groupby("patient_id").size().describe().round(2).to_string(),
    "", "Image mode:", df["mode"].value_counts().to_string(),
    "", "Width:", df.width.describe().round(0).to_string(),
    "", "Height:", df.height.describe().round(0).to_string(),
    "", "Split x label (images):", pd.crosstab(df.split, df.label, margins=True).to_string(),
    "", "Split x label (patients):", df.groupby(["split", "label"]).patient_id.nunique().unstack().to_string(),
    "", "Original Kaggle folder x label:", pd.crosstab(df.kaggle_split, df.label).to_string(),
    "", f"Patient overlap test vs dev: {len(overlap)}",
]
(OUT / "profile_summary.txt").write_text("\n".join(lines))
print("\n".join(lines))
