"""
Step 7: Descriptive statistics for the real chest X-ray dataset (and a synthetic sample).
Run from D:\\School\\synthetic_data:   python scripts\\07_descriptive_stats.py
Reads outputs\\step1\\manifest.csv and outputs\\step7\\image_brightness.csv
Writes outputs\\step7\\descriptive_stats.csv  (one row per measure; columns NORMAL, PNEUMONIA, TOTAL)
Brightness/contrast = mean / SD of pixel values (0-255) in the central chest region after padding to square.
"""
from pathlib import Path
import numpy as np, pandas as pd, cv2
from PIL import Image

R = Path(__file__).resolve().parents[1]
m = pd.read_csv(R / "outputs/step1/manifest.csv")
b = pd.read_csv(R / "outputs/step7/image_brightness.csv")
N, P = m[m.label == "NORMAL"], m[m.label == "PNEUMONIA"]
bN, bP = b[b.label == "NORMAL"], b[b.label == "PNEUMONIA"]
ppat = lambda d: d.groupby("patient_id").size()
k = pd.read_csv(R / "outputs/step1/manifest.csv")
tr_p, te_p = set(k[k.kaggle_split == "train"].patient_id), set(k[k.kaggle_split == "test"].patient_id)
overlap = tr_p & te_p
leaked = k[(k.kaggle_split == "test") & k.patient_id.isin(overlap)]
rng = lambda s: f"{s.min():.0f}-{s.max():.0f}"
msd = lambda s: f"{s.mean():.1f} ({s.std():.1f})"

# synthetic sample: 40 images per class from each of seeds 0, 4, 7, 9 (n = 160 per class)
cl = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)); syn = {}
for lab in ("NORMAL", "PNEUMONIA"):
    v = []
    for s in (0, 4, 7, 9):
        sm = pd.read_csv(R / f"outputs/step4/synthetic/seed_{s}/synthetic_manifest.csv")
        for f in sm[(sm.kept == 1) & (sm.label == lab)].sample(40, random_state=0).file:
            a = cv2.resize(np.asarray(Image.open(R / f"outputs/step4/synthetic/seed_{s}/{f}").convert("L")), (256, 256))
            c, cc = a[64:192, 48:208].astype(float), cl.apply(a)[64:192, 48:208].astype(float)
            v.append((c.mean(), c.std(), cc.mean(), cc.std()))
    syn[lab] = np.array(v)

rows = [
    ("unique_images", len(N), len(P), len(m)),
    ("bacterial_images", "", (P.subtype == "bacteria").sum(), (P.subtype == "bacteria").sum()),
    ("viral_images", "", (P.subtype == "virus").sum(), (P.subtype == "virus").sum()),
    ("exact_duplicates_removed_by_hash", "", "", 32),
    ("patients", N.patient_id.nunique(), P.patient_id.nunique(), m.patient_id.nunique()),
    ("images_per_patient_mean", round(ppat(N).mean(), 2), round(ppat(P).mean(), 2), round(ppat(m).mean(), 2)),
    ("images_per_patient_median", ppat(N).median(), ppat(P).median(), ppat(m).median()),
    ("images_per_patient_max", ppat(N).max(), ppat(P).max(), ppat(m).max()),
    ("class_ratio_pneumonia_to_normal", "", "", round(len(P) / len(N), 2)),
    ("kaggle_patients_in_train_and_test", "", "", len(overlap)),
    ("kaggle_test_images_from_those_patients", "", "", f"{len(leaked)} of {(k.kaggle_split=='test').sum()} ({len(leaked)/(k.kaggle_split=='test').sum():.1%})"),
    ("width_px_mean", round(N.width.mean()), round(P.width.mean()), round(m.width.mean())),
    ("width_px_range", rng(N.width), rng(P.width), rng(m.width)),
    ("height_px_mean", round(N.height.mean()), round(P.height.mean()), round(m.height.mean())),
    ("height_px_range", rng(N.height), rng(P.height), rng(m.height)),
    ("grayscale_images", (N["mode"] == "L").sum(), (P["mode"] == "L").sum(), (m["mode"] == "L").sum()),
    ("color_rgb_images", (N["mode"] == "RGB").sum(), (P["mode"] == "RGB").sum(), (m["mode"] == "RGB").sum()),
    ("model_input_after_preprocessing", "224x224 grayscale", "224x224 grayscale", "224x224 grayscale"),
    ("chest_brightness_mean_sd", msd(bN.chest_mean), msd(bP.chest_mean), msd(b.chest_mean)),
    ("chest_brightness_range", rng(bN.chest_mean), rng(bP.chest_mean), rng(b.chest_mean)),
    ("chest_contrast_mean", round(bN.chest_std.mean(), 1), round(bP.chest_std.mean(), 1), round(b.chest_std.mean(), 1)),
    ("clahe_chest_brightness_mean_sd", msd(bN.chest_mean_clahe), msd(bP.chest_mean_clahe), msd(b.chest_mean_clahe)),
    ("clahe_chest_brightness_range", rng(bN.chest_mean_clahe), rng(bP.chest_mean_clahe), rng(b.chest_mean_clahe)),
    ("clahe_chest_contrast_mean", round(bN.chest_std_clahe.mean(), 1), round(bP.chest_std_clahe.mean(), 1), round(b.chest_std_clahe.mean(), 1)),
    ("synthetic_sample_n", len(syn["NORMAL"]), len(syn["PNEUMONIA"]), len(syn["NORMAL"]) + len(syn["PNEUMONIA"])),
    ("synthetic_chest_brightness_mean", round(syn["NORMAL"][:, 0].mean(), 1), round(syn["PNEUMONIA"][:, 0].mean(), 1), ""),
    ("synthetic_chest_contrast_mean", round(syn["NORMAL"][:, 1].mean(), 1), round(syn["PNEUMONIA"][:, 1].mean(), 1), ""),
    ("synthetic_clahe_chest_brightness_mean", round(syn["NORMAL"][:, 2].mean(), 1), round(syn["PNEUMONIA"][:, 2].mean(), 1), ""),
    ("synthetic_clahe_chest_contrast_mean", round(syn["NORMAL"][:, 3].mean(), 1), round(syn["PNEUMONIA"][:, 3].mean(), 1), ""),
]
df = pd.DataFrame(rows, columns=["measure", "NORMAL", "PNEUMONIA", "TOTAL"]).set_index("measure")
df.to_csv(R / "outputs/step7/descriptive_stats.csv")
pd.set_option("display.width", 200); print(df.to_string())
