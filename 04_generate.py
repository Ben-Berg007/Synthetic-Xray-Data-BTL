"""
Step 4: Generate the synthetic training set for one seed from its chosen LoRA checkpoint,
and drop any image that is a near-copy of that seed's 11 real training images.

Checkpoint rule (fixed on seed 0, 2026-09-27, applied unchanged to seeds 1-4):
    500 steps = highest class_gap with zero near-copies (see lora_seed_0_v2/checkpoint_scores.csv)

Run from D:\\School\\synthetic_data:
    python scripts\\04_generate.py --seed 0
Outputs: outputs\\step4\\synthetic\\seed_k\\{NORMAL,PNEUMONIA}\\*.png  (512x512 grayscale)
         outputs\\step4\\synthetic\\seed_k\\synthetic_manifest.csv   (file, label, gen index, nn_sim, kept)
"""
import argparse, time
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
CHECKPOINT_STEP = 500
NN_THRESHOLD = 0.95
ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--tag", default="v2")
ap.add_argument("--n_per_class", type=int, default=1000)
ap.add_argument("--batch", type=int, default=8)
args = ap.parse_args()
dev = "cuda"

ck = ROOT / "outputs" / "step3" / f"lora_seed_{args.seed}_{args.tag}" / f"checkpoint_{CHECKPOINT_STEP}"
assert ck.exists(), f"Missing {ck} - train this seed first"
out = ROOT / "outputs" / "step4" / "synthetic" / f"seed_{args.seed}"

# feature extractor for the near-copy check (same as 03b)
inc = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1).to(dev).eval(); inc.fc = torch.nn.Identity()
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
@torch.no_grad()
def feats(pils):
    x = torch.stack([torch.from_numpy(np.asarray(p.convert("L").resize((299, 299)), dtype=np.float32) / 255.)
                     for p in pils]).unsqueeze(1).repeat(1, 3, 1, 1).to(dev)
    return F.normalize(inc((x - MEAN) / STD).float(), dim=1)
fs = pd.read_csv(ROOT / "outputs" / "step2" / "fewshot_sets.csv"); fs = fs[fs.seed == args.seed]
train_f = feats([ImageOps.pad(Image.open(ROOT / p).convert("L"), (512, 512), color=0) for p in fs.path])

pipe = StableDiffusionPipeline.from_pretrained(MODEL, torch_dtype=torch.float16, variant="fp16",
                                               safety_checker=None, requires_safety_checker=False).to(dev)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.load_lora_weights(ck); pipe.set_progress_bar_config(disable=True)

rows, t0 = [], time.time()
for ci, (lab, prompt) in enumerate(PROMPTS.items()):
    (out / lab).mkdir(parents=True, exist_ok=True)
    g = torch.Generator(dev).manual_seed(100000 * (args.seed + 1) + ci)   # distinct from 03b's eval seed
    kept = i = 0
    while kept < args.n_per_class:
        ims = pipe([prompt] * args.batch, num_inference_steps=25, guidance_scale=5.0, generator=g).images
        sims = (feats(ims) @ train_f.T).max(dim=1).values.tolist()
        for im, s in zip(ims, sims):
            ok = s <= NN_THRESHOLD and kept < args.n_per_class
            name = f"{lab.lower()}_{i:05d}.png"
            if ok:
                im.convert("L").save(out / lab / name); kept += 1
            rows.append(dict(file=f"{lab}/{name}" if ok else "", label=lab, y=int(lab == "PNEUMONIA"),
                             seed=args.seed, gen_index=i, nn_sim=round(s, 4), kept=int(ok)))
            i += 1
        if i % 200 < args.batch:
            print(f"seed {args.seed} {lab}: {kept}/{args.n_per_class} kept  ({(time.time()-t0)/60:.1f} min)")
df = pd.DataFrame(rows); df.to_csv(out / "synthetic_manifest.csv", index=False)
rej = (df.kept == 0) & (df.nn_sim > NN_THRESHOLD)
print(f"Done seed {args.seed}: kept {df.kept.sum()}, rejected as near-copies {rej.sum()}, "
      f"max nn_sim {df.nn_sim.max():.3f}, {(time.time()-t0)/60:.1f} min")
