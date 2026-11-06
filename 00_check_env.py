"""Quick environment check. Run: python scripts\\00_check_env.py"""
import importlib, sys
print("Python", sys.version.split()[0])
for m in ["torch", "torchvision", "diffusers", "transformers", "peft", "accelerate", "pandas", "sklearn", "PIL"]:
    try:
        mod = importlib.import_module(m); print(f"{m:13s} {getattr(mod, '__version__', 'ok')}")
    except Exception as e:
        print(f"{m:13s} MISSING ({e.__class__.__name__})")
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("GPU:", p.name, f"| VRAM {p.total_memory/1024**3:.1f} GB | CUDA {torch.version.cuda}")
    x = torch.randn(2048, 2048, device="cuda", dtype=torch.float16); print("fp16 matmul ok:", (x @ x).shape)
