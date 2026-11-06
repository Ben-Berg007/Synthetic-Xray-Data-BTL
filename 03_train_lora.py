"""
Step 3: LoRA fine-tune Stable Diffusion 1.5 on ONE seed's 11 real X-rays (few-shot generator).

Run from D:\\School\\synthetic_data (one seed at a time):
    python scripts\\03_train_lora.py --seed 0
    python scripts\\03_train_lora.py --seed 0 --steps 1500 --rank 16      (overrides)

Inputs : outputs\\step2\\fewshot\\seed_k\\  (11 PNGs + metadata.jsonl)
Outputs: outputs\\step3\\lora_seed_k\\
           checkpoint_<step>\\pytorch_lora_weights.safetensors   (every --save_every steps)
           preview_<step>.png   4 NORMAL (top row) + 4 PNEUMONIA (bottom row) samples
           loss_log.csv, run_config.json

Memory plan for a 12 GB RTX 3060: frozen VAE/text encoder/UNet in fp16, LoRA weights in fp32,
autocast + GradScaler, gradient checkpointing, batch size 2. Expect ~7-9 GB and ~15-25 min per 1,500 steps.
No horizontal flips: left/right anatomy (heart position) matters in chest X-rays.
v2 (2026-09-27): removed RandomRotation - the seed-0 run reproduced the rotated/grey frame
artifacts it created. Default 1,000 steps with checkpoints every 250 (1,500 over-sharpened ribs
and weakened the pneumonia/normal difference).
"""
import argparse, csv, json, random, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from diffusers import (AutoencoderKL, DDPMScheduler, DPMSolverMultistepScheduler,
                       StableDiffusionPipeline, UNet2DConditionModel)
from diffusers.utils import convert_state_dict_to_diffusers
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from transformers import CLIPTextModel, CLIPTokenizer

ROOT = Path(__file__).resolve().parents[1]
MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
PROMPTS = {"NORMAL": "a chest x-ray, normal lungs", "PNEUMONIA": "a chest x-ray, pneumonia"}

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--steps", type=int, default=1000)
ap.add_argument("--save_every", type=int, default=250)
ap.add_argument("--rank", type=int, default=16)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--batch", type=int, default=2)
ap.add_argument("--res", type=int, default=512)
ap.add_argument("--caption_dropout", type=float, default=0.1)
ap.add_argument("--tag", default="", help="suffix for the output folder, e.g. v2")
args = ap.parse_args()

assert torch.cuda.is_available(), "CUDA GPU not found - check the PyTorch CUDA install (run scripts\\00_check_env.py)"
dev = "cuda"
random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

data_dir = ROOT / "outputs" / "step2" / "fewshot" / f"seed_{args.seed}"
out_dir = ROOT / "outputs" / "step3" / (f"lora_seed_{args.seed}" + (f"_{args.tag}" if args.tag else ""))
out_dir.mkdir(parents=True, exist_ok=True)
meta = [json.loads(l) for l in open(data_dir / "metadata.jsonl")]
print(f"Seed {args.seed}: {len(meta)} training images from {data_dir}")

# ---------------- models ----------------
wdt = torch.float16
tok = CLIPTokenizer.from_pretrained(MODEL, subfolder="tokenizer")
te = CLIPTextModel.from_pretrained(MODEL, subfolder="text_encoder", torch_dtype=wdt, variant="fp16").to(dev)
vae = AutoencoderKL.from_pretrained(MODEL, subfolder="vae", torch_dtype=wdt, variant="fp16").to(dev)
unet = UNet2DConditionModel.from_pretrained(MODEL, subfolder="unet", torch_dtype=wdt, variant="fp16").to(dev)
noise_sched = DDPMScheduler.from_pretrained(MODEL, subfolder="scheduler")
for m in (te, vae, unet):
    m.requires_grad_(False)

unet.add_adapter(LoraConfig(r=args.rank, lora_alpha=args.rank, init_lora_weights="gaussian",
                            target_modules=["to_k", "to_q", "to_v", "to_out.0"]))
for p in unet.parameters():                     # train LoRA weights in fp32
    if p.requires_grad:
        p.data = p.data.float()
unet.enable_gradient_checkpointing()
unet.train()
params = [p for p in unet.parameters() if p.requires_grad]
print(f"Trainable LoRA parameters: {sum(p.numel() for p in params)/1e6:.2f} M")
opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-2)
scaler = torch.amp.GradScaler("cuda")

@torch.no_grad()
def embed(texts):
    ids = tok(texts, padding="max_length", max_length=tok.model_max_length,
              truncation=True, return_tensors="pt").input_ids.to(dev)
    return te(ids)[0]
emb = {k: embed([v]) for k, v in PROMPTS.items()}
emb[""] = embed([""])

# ---------------- data (light, anatomy-preserving augmentation) ----------------
aug = transforms.Compose([
    transforms.RandomResizedCrop(args.res, scale=(0.85, 1.0), ratio=(0.95, 1.05)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),                       # 1 x H x W in [0,1]
])
imgs = [(Image.open(data_dir / m["file_name"]).convert("L"), m["label"]) for m in meta]

def batch():
    xs, cs = [], []
    for im, lab in random.choices(imgs, k=args.batch):
        x = aug(im).repeat(3, 1, 1) * 2 - 1      # grayscale -> 3ch, [-1, 1]
        xs.append(x)
        cs.append(emb[""] if random.random() < args.caption_dropout else emb[lab])
    return torch.stack(xs).to(dev, wdt), torch.cat(cs)

# ---------------- preview ----------------
@torch.no_grad()
def preview(step):
    unet.eval()
    pipe = StableDiffusionPipeline(vae=vae, text_encoder=te, tokenizer=tok, unet=unet,
                                   scheduler=DPMSolverMultistepScheduler.from_config(noise_sched.config),
                                   safety_checker=None, feature_extractor=None, requires_safety_checker=False)
    pipe.set_progress_bar_config(disable=True)
    g = torch.Generator(dev).manual_seed(1234)
    rows = []
    for lab in ("NORMAL", "PNEUMONIA"):
        with torch.autocast("cuda", dtype=wdt):
            ims = pipe([PROMPTS[lab]] * 4, num_inference_steps=25, guidance_scale=5.0, generator=g,
                       height=args.res, width=args.res).images
        rows.append(ims)
    s = 256
    grid = Image.new("L", (4 * s, 2 * s))
    for r, ims in enumerate(rows):
        for c, im in enumerate(ims):
            grid.paste(im.convert("L").resize((s, s)), (c * s, r * s))
    grid.save(out_dir / f"preview_{step}.png")
    unet.train()

def save(step):
    d = out_dir / f"checkpoint_{step}"
    StableDiffusionPipeline.save_lora_weights(
        d, unet_lora_layers=convert_state_dict_to_diffusers(get_peft_model_state_dict(unet)),
        safe_serialization=True)
    preview(step)
    print(f"  saved {d.name} + preview_{step}.png")

# ---------------- train ----------------
json.dump({**vars(args), "model": MODEL, "prompts": PROMPTS, "n_images": len(meta)},
          open(out_dir / "run_config.json", "w"), indent=2)
log = open(out_dir / "loss_log.csv", "w", newline=""); lw = csv.writer(log); lw.writerow(["step", "loss"])
t0, run = time.time(), 0.0
for step in range(1, args.steps + 1):
    px, cond = batch()
    with torch.no_grad():
        lat = vae.encode(px).latent_dist.sample() * vae.config.scaling_factor
    noise = torch.randn_like(lat)
    t = torch.randint(0, noise_sched.config.num_train_timesteps, (lat.shape[0],), device=dev)
    noisy = noise_sched.add_noise(lat, noise, t)
    with torch.autocast("cuda", dtype=wdt):
        pred = unet(noisy, t, cond).sample
    loss = F.mse_loss(pred.float(), noise.float())
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 1.0)
    scaler.step(opt); scaler.update()
    run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
    lw.writerow([step, round(loss.item(), 5)])
    if step % 50 == 0:
        el = time.time() - t0
        print(f"step {step}/{args.steps}  loss(ema) {run:.4f}  {el/step:.2f}s/step  "
              f"eta {el/step*(args.steps-step)/60:.1f} min  vram {torch.cuda.max_memory_allocated()/1024**3:.1f} GB")
    if step % args.save_every == 0 or step == args.steps:
        save(step)
log.close()
print(f"Done in {(time.time()-t0)/60:.1f} min -> {out_dir}")
