# Restore-Physics Project

This project focuses on physics-informed speckle restoration with a Swin-UNet denoiser and frequency-aware losses.

## What It Does

- domain A: speckle images
- domain B: clean / ground-truth images
- preprocessing: histogram equalization on the speckle domain before training and inference
- model: physics-informed Swin-UNet denoiser

## Dataset Layout

Use your predefined split structure:

```text
output/
  train/
    speckle/
      video_name/frame.png
    groundtruth/
      video_name/frame.png
  val/
    speckle/
      video_name/frame.png
    groundtruth/
      video_name/frame.png
  test/
    speckle/
      video_name/frame.png
    groundtruth/
      video_name/frame.png
```

Training and validation are paired (`speckle` ↔ `groundtruth`).

## Speckle Data preperation

```bash
python3 distorter.py 
```

## Train

```bash
python3 -m restore_physics.train_physics --config configs/restore_physics.yaml --device cuda:0
```

## Inference

```bash
python3 -m restore_physics.infer \
  --checkpoint outputs_restorephysics/restore_physics_swinunet_denoiser/best.pt \
  --input path/to/speckle.png \
  --output path/to/restored.png \
  --use-ema
```

Inference expects a Swin-UNet physics checkpoint (`model_name: swin_unet_physics_denoiser`).

## Stability Features

- mixed precision training (`runtime.amp`)
- gradient accumulation (`train.stability.grad_accum_steps`)
- cosine LR with warmup (`warmup_steps`, `min_lr_ratio`)
- EMA model averaging (`use_ema`, `ema_decay`)
- gradient clipping (`train.grad_clip_norm`)

## Physics Priors

Configured under `train` and `train.physics_loss`:

- `lambda_log_noise_mean`: log-noise zero-mean prior
- `lambda_log_noise_tv`: spatial smoothness prior on log-noise
- `lambda_gamma_nll`: multi-scale Gamma speckle NLL prior weight
- `lambda_enl`: ENL consistency prior weight (homogeneous regions)
- `eps`: numerical stability term
- `frc_rings`: radial bins for FRC loss / weighted FRC loss
- `gamma_looks`: equivalent number of looks in Gamma prior
- `gamma_scales`: scales used for multi-scale Gamma prior (for example `[1,2,4]`)
- `expected_enl`: target ENL for homogeneous regions
- `enl_patch_size`: local patch size for ENL estimation
- `enl_homogeneity_cv_max`: CV threshold to detect homogeneous patches
- `use_weighted_frc`: enable ring-weighted FRC
- `frc_low_freq_weight`: weight near low-frequency rings
- `frc_high_freq_weight`: weight near high-frequency rings

## Logged Metrics

- `train_loss`
- `train_charbonnier_loss`
- `train_grad_loss`
- `train_frc_loss`
- `train_smooth_loss`
- `train_log_noise_mean_loss`
- `train_log_noise_tv_loss`
- `train_gamma_nll_loss`
- `train_enl_loss`
- `val_l1_loss`
- `val_psnr`
- `val_ssim`
- `val_frc_score`
