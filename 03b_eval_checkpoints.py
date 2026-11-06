"""
Step 3b: Score every LoRA checkpoint for one seed so the checkpoint rule is chosen by numbers,
not by eye. Run AFTER 03_train_lora.py, e.g.:
    python scripts\\03b_eval_checkpoints.py --seed 0 --tag v2

For each checkpoint it generates --n images per class (fixed generator seed) and reports:
  KID (x1000)   Kernel Inception Distance vs real dev_train images of the same class
                (lower = closer to real). Uses up to --n_real real images per class, never the test set.
  class_gap     KID(synthetic pneumonia vs real NORMAL) - KID(synthetic pneumonia vs real PNEUMONIA)
                (higher = generated pneumonia looks more like pneumonia than like normal)
  nn_sim_max    highest cosine similarity between any synthetic image and any of the 11 training
                images (Inception features); nn_frac_095 = share of synthetic images above 0.95
                (higher = closer to copying the training images = memorization/privacy risk)
Outputs: outputs\\step3\\lora_seed_k[_tag]\\checkpoint_scores.csv  (+ a printed table)
Real-image features are cached in outputs\\step3\\real_feats_seed_k.pt.
Note: using extra real images here is a researcher-side diagnostic to set ONE hyperparameter
(step count) on seed 0; it is then fixed for seeds 1-4 and disclosed as a limitation.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision.models import inception_v3, Inception_V3_Weights
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

ROOT = Path(__file__).resolve().parents[1]
MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
PROMPTS = {"NORMAL": "a chest x-ray, normal lungs", "PNEUMONIA": "a chest x-ray, pneumonia"}
ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--tag", default="")
ap.add_argument("--n", type=int, default=100, help="synthetic images per class per checkpoint")
ap.add_argument("--n_real", type=int, default=300)
args = ap.parse_args()
dev = "cuda"
run_dir = ROOT / "outputs" / "step3" / (f"lora_seed_{args.seed}" + (f"_{args.tag}" if args.tag else ""))

# ---------- feature extractor (Inception-v3 pool, 2048-d) ----------
w = Inception_V3_Weights.IMAGENET1K_V1
inc = inception_v3(weights=w).to(dev).eval(); inc.fc = torch.nn.Identity()
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

@torch.no_grad()
def feats(pils, bs=32):
    out = []
    for i in range(0, len(pils), bs):
        x = torch.stack([torch.from_numpy(np.asarray(p.convert("L").resize((299, 299)), dtype=np.float32) / 255.)
                         for p in pils[i:i + bs]]).unsqueeze(1).repeat(1, 3, 1, 1).to(dev)
        out.append(inc((x - MEAN) / STD).float())
    return torch.cat(out)

def load_real(path):
    return ImageOps.pad(Image.open(ROOT / path).convert("L"), (512, 512), color=0)

def kid(a, b, subsets=20, size=None):
    a, b = a.double(), b.double(); d = a.shape[1]
    size = size or min(len(a), len(b), 100); vals = []
    g = torch.Generator().manual_seed(0)
    for _ in range(subsets):
        x = a[torch.randperm(len(a), generator=g)[:size]]; y = b[torch.randperm(len(b), generator=g)[:size]]
        k = lambda u, v: (u @ v.T / d + 1) ** 3
        kxx, kyy, kxy = k(x, x), k(y, y), k(x, y); m = size
        vals.append(((kxx.sum() - kxx.diag().sum()) / (m * (m - 1)) + (kyy.sum() - kyy.diag().sum()) / (m * (m - 1))
                     - 2 * kxy.mean()).item())
    return float(np.mean(vals)) * 1000

# ---------- real reference features (dev_train only, excluding this seed's few-shot images) ----------
cache = ROOT / "outputs" / "step3" / f"real_feats_seed_{args.seed}.pt"
fs = pd.read_csv(ROOT / "outputs" / "step2" / "fewshot_sets.csv"); fs = fs[fs.seed == args.seed]
if cache.exists():
    real = torch.load(cache)
else:
    asg = pd.read_csv(ROOT / "outputs" / "step2" / "assignments.csv")
    pool = asg[(asg.role == "dev_train") & (~asg.md5.isin(fs.md5))]
    real = {}
    for lab in PROMPTS:
        paths = pool[pool.label == lab].sample(n=args.n_real, random_state=0).path
        real[lab] = feats([load_real(p) for p in paths]).cpu()
    real["train11"] = feats([load_real(p) for p in fs.path]).cpu()
    torch.save(real, cache)
tr = F.normalize(real["train11"], dim=1)

# ---------- score checkpoints ----------
pipe = StableDiffusionPipeline.from_pretrained(MODEL, torch_dtype=torch.float16, variant="fp16",
                                               safety_checker=None, requires_safety_checker=False).to(dev)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.set_progress_bar_config(disable=True)
ckpts = sorted(run_dir.glob("checkpoint_*"), key=lambda p: int(p.name.split("_")[1]))
rows = []
for ck in ckpts:
    pipe.unload_lora_weights(); pipe.load_lora_weights(ck)
    syn = {}
    for lab, prompt in PROMPTS.items():
        g = torch.Generator(dev).manual_seed(777); ims = []
        for i in range(0, args.n, 8):
            ims += pipe([prompt] * min(8, args.n - i), num_inference_steps=25, guidance_scale=5.0, generator=g).images
        syn[lab] = feats(ims).cpu()
    allsyn = F.normalize(torch.cat([syn["NORMAL"], syn["PNEUMONIA"]]), dim=1)
    nn = (allsyn @ tr.T).max(dim=1).values
    rows.append(dict(checkpoint=int(ck.name.split("_")[1]),
                     kid_normal=kid(syn["NORMAL"], real["NORMAL"]),
                     kid_pneumonia=kid(syn["PNEUMONIA"], real["PNEUMONIA"]),
                     class_gap=kid(syn["PNEUMONIA"], real["NORMAL"]) - kid(syn["PNEUMONIA"], real["PNEUMONIA"]),
                     nn_sim_max=nn.max().item(), nn_sim_mean=nn.mean().item(),
                     nn_frac_095=(nn > 0.95).float().mean().item()))
    print(rows[-1])
df = pd.DataFrame(rows).round(4)
df.to_csv(run_dir / "checkpoint_scores.csv", index=False)
print("\n" + df.to_string(index=False))
