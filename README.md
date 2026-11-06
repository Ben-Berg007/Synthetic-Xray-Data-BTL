# Synthetic Augmentation to Bridge the Data Gap in Chest X-Ray Classification

MIS581 Capstone, Colorado State University Global. Author: Benjamin Bergenstein.

Can synthetic chest X-rays help a small healthcare organization that holds only 11 labeled real images? This project fine-tunes Stable Diffusion 1.5 with LoRA on 11 real X-rays, generates synthetic X-rays, and compares four ways of training a ResNet-18 pneumonia classifier on one patient-level hold-out test set:

| Condition | Training data |
|---|---|
| Real-only (full) | 3,896 real training images (benchmark) |
| Real-only (11) | The same 11 real images used by the generator |
| Synthetic-only | 2,000 synthetic images, no real data |
| BTL (bridged transfer learning) | Pretrain on the 2,000 synthetic images, then fine-tune on the 11 real images |

Each condition is repeated over 10 random draws of 11 images (seeds 0–9).

## Key results (final run, mean of 10 draws)

| Metric | Real-only (full) | Real-only (11) | Synthetic-only | BTL |
|---|---|---|---|---|
| MCC (primary, threshold 0.5) | 0.502 | 0.272 | 0.224 | 0.328 |
| ROC-AUC | 0.953 | 0.784 | 0.810 | 0.870 |
| MCC (validation-calibrated threshold) | 0.746 | 0.383 | 0.457 | 0.543 |
| Sensitivity | 0.967 | 0.838 | 0.931 | 0.871 |

- **H1 rejected:** synthetic-only scored significantly lower MCC than full real data (−0.277, Holm-adjusted p = .024).
- **H2a, H2b not rejected:** BTL did not significantly improve MCC at the default threshold. BTL did significantly improve ROC-AUC (+0.087) and calibrated MCC (+0.160) over the 11-image model.
- **H3 not rejected:** BTL ROC-AUC was 0.083 below full real data, 95% CI [−0.107, −0.058], so the gap is only partially closed.
- All decisions were unchanged under a patient-level bootstrap (`06c_patient_bootstrap.py`).

![Per-seed test performance](figures/metrics_by_condition_v2.png)

## Pipeline

| Step | Script | What it does |
|---|---|---|
| 0 | `00_check_env.py` | Checks Python, GPU, and library versions |
| 1 | `01_profile_and_split.py` | Removes duplicates, finds the patient leakage in the original Kaggle split, creates the patient-level hold-out test set first |
| 2 | `02_draw_subsets.py` | Carves out the validation partition, draws the 10 seeded sets of 11 images (5 normal, 6 pneumonia, one per patient) |
| 3 | `03_train_lora.py`, `03b_eval_checkpoints.py` | LoRA fine-tunes Stable Diffusion 1.5 on each seed's 11 images; scores checkpoints (KID, class gap, near-copy similarity) |
| 4 | `04_generate.py` | Generates 1,000 images per class per seed, rejects near-copies (cosine > 0.95) |
| 5 | `05_train_classifiers.py` (run 1), `05b_fewshot_v2.py` (run 2, final) | Trains the four classifier conditions |
| 6 | `06_stats.py` (run 1), `06b_stats_v2.py` (run 2, final), `06c_patient_bootstrap.py` | Paired bootstrap CIs, paired t, Wilcoxon, McNemar, Holm correction, gap-closure tiers; patient-level sensitivity analysis |
| 7 | `07_descriptive_stats.py` | Descriptive statistics and the CLAHE before/after figure |

`run_remaining_seeds.bat` and `run_seeds_5_9.bat` run steps 3–4 for multiple seeds on Windows.

Run 1 is kept as a documented first attempt (the few-shot models overfit and predicted pneumonia for nearly every image). Run 2 is the final, pre-registered protocol; its training recipe was selected on the validation partition only.

## Reproducing

1. Install PyTorch with CUDA from https://pytorch.org/get-started/locally/, then `pip install -r scripts/requirements.txt`.
2. Download the Chest X-Ray Images (Pneumonia) dataset and place it under `data/chest_xray/` (see the paths in `01_profile_and_split.py`).
3. Run the scripts in order from the project root. Seeds and random states are fixed in each script.

Hardware used: Windows, NVIDIA RTX 3060 (12 GB).

## Repository contents

```
scripts/   all pipeline code and requirements.txt
results/   small result files: profiling, seed draws, generator QC, synthetic manifests,
           training metrics, predictions, and statistics for run 1, run 2, and the patient-level bootstrap
figures/   figures used in the paper and presentation
docs/      final paper and presentation
```

Not included (size and licensing): the X-ray images, the 20,000 synthetic images, the LoRA weights, and the trained classifier weights. Everything can be regenerated with the scripts.

## Data and model licenses

- Dataset: Kermany, D., Zhang, K., & Goldbaum, M. (2018). *Labeled optical coherence tomography (OCT) and chest X-ray images for classification* (Version 2) [Data set]. Mendeley Data. https://doi.org/10.17632/rscbjbr9sj.2 (CC BY 4.0). Kaggle distribution: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
- Generator: Stable Diffusion 1.5, https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5 (CreativeML OpenRAIL-M license).

These models are research models and are not intended for clinical use.
