from __future__ import annotations

import argparse
import contextlib
import copy
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from config import Config
from data import create_paired_train_dataloader, create_validation_dataloader
from models import SwinUNetDenoiser
from utils import (
    append_metrics,
    compute_psnr,
    compute_ssim,
    ensure_dir,
    save_preview,
    set_seed,
    utc_timestamp,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train physics-informed Swin-UNet denoiser for speckle restoration")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--device", type=str, help="Override device, for example cuda:0")
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def to_unit_range(x: torch.Tensor) -> torch.Tensor:
    return (x + 1.0) * 0.5


def configure_cuda_runtime(runtime_cfg: dict, device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.backends.cudnn.benchmark = bool(runtime_cfg.get("cudnn_benchmark", True))
    torch.backends.cuda.matmul.allow_tf32 = bool(runtime_cfg.get("allow_tf32", True))
    torch.backends.cudnn.allow_tf32 = bool(runtime_cfg.get("allow_tf32", True))


def charbonnier_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt((pred - target) ** 2 + eps**2).mean()


def total_variation_loss(x: torch.Tensor) -> torch.Tensor:
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    return dx.abs().mean() + dy.abs().mean()


def gradient_map(x: torch.Tensor) -> torch.Tensor:
    gx = x[:, :, :, 1:] - x[:, :, :, :-1]
    gy = x[:, :, 1:, :] - x[:, :, :-1, :]
    gx = torch.nn.functional.pad(gx, (0, 1, 0, 0), mode="replicate")
    gy = torch.nn.functional.pad(gy, (0, 0, 0, 1), mode="replicate")
    return torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-8)


def radial_bins(h: int, w: int, num_rings: int, device: torch.device) -> torch.Tensor:
    yy = torch.linspace(-1.0, 1.0, h, device=device).view(h, 1).expand(h, w)
    xx = torch.linspace(-1.0, 1.0, w, device=device).view(1, w).expand(h, w)
    rr = torch.sqrt(xx**2 + yy**2).clamp(max=1.0)
    return torch.clamp((rr * num_rings).long(), min=0, max=num_rings - 1)


def frc_loss(pred_01: torch.Tensor, target_01: torch.Tensor, num_rings: int = 32, eps: float = 1e-8) -> torch.Tensor:
    b, c, h, w = pred_01.shape
    bins = radial_bins(h, w, num_rings=num_rings, device=pred_01.device).reshape(-1)

    pred_fft = torch.fft.fftshift(torch.fft.fft2(pred_01, norm="ortho"), dim=(-2, -1))
    target_fft = torch.fft.fftshift(torch.fft.fft2(target_01, norm="ortho"), dim=(-2, -1))

    corr = (pred_fft * torch.conj(target_fft)).real.reshape(b, c, -1)
    p_pow = (pred_fft.abs() ** 2).reshape(b, c, -1)
    t_pow = (target_fft.abs() ** 2).reshape(b, c, -1)

    frc_scores = []
    for ring in range(num_rings):
        mask = bins == ring
        if not mask.any():
            continue
        num = corr[:, :, mask].sum(dim=-1)
        den = torch.sqrt(p_pow[:, :, mask].sum(dim=-1) * t_pow[:, :, mask].sum(dim=-1) + eps)
        frc_scores.append((num / den).clamp(min=-1.0, max=1.0).mean())

    if not frc_scores:
        return pred_01.new_tensor(0.0)
    return 1.0 - torch.stack(frc_scores).mean()


def weighted_frc_loss(
    pred_01: torch.Tensor,
    target_01: torch.Tensor,
    num_rings: int = 32,
    low_freq_weight: float = 1.0,
    high_freq_weight: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    b, c, h, w = pred_01.shape
    bins = radial_bins(h, w, num_rings=num_rings, device=pred_01.device).reshape(-1)

    pred_fft = torch.fft.fftshift(torch.fft.fft2(pred_01, norm="ortho"), dim=(-2, -1))
    target_fft = torch.fft.fftshift(torch.fft.fft2(target_01, norm="ortho"), dim=(-2, -1))

    corr = (pred_fft * torch.conj(target_fft)).real.reshape(b, c, -1)
    p_pow = (pred_fft.abs() ** 2).reshape(b, c, -1)
    t_pow = (target_fft.abs() ** 2).reshape(b, c, -1)

    weighted_terms = []
    weights = []
    for ring in range(num_rings):
        mask = bins == ring
        if not mask.any():
            continue
        num = corr[:, :, mask].sum(dim=-1)
        den = torch.sqrt(p_pow[:, :, mask].sum(dim=-1) * t_pow[:, :, mask].sum(dim=-1) + eps)
        frc_ring = (num / den).clamp(min=-1.0, max=1.0).mean()
        radius = ring / max(num_rings - 1, 1)
        weight = low_freq_weight * (1.0 - radius) + high_freq_weight * radius
        weighted_terms.append((1.0 - frc_ring) * weight)
        weights.append(weight)

    if not weighted_terms:
        return pred_01.new_tensor(0.0)
    return torch.stack(weighted_terms).sum() / max(sum(weights), eps)


def gamma_speckle_nll(noise_ratio: torch.Tensor, looks: float, eps: float) -> torch.Tensor:
    k = max(float(looks), eps)
    n = noise_ratio.clamp(min=eps)
    return (k * n - (k - 1.0) * torch.log(n)).mean()


def multiscale_gamma_speckle_nll(
    input_01: torch.Tensor,
    pred_01: torch.Tensor,
    looks: float,
    eps: float,
    scales: tuple[int, ...],
) -> torch.Tensor:
    losses = []
    for scale in scales:
        if scale <= 1:
            in_s = input_01
            pr_s = pred_01
        else:
            in_s = torch.nn.functional.avg_pool2d(input_01, kernel_size=scale, stride=scale)
            pr_s = torch.nn.functional.avg_pool2d(pred_01, kernel_size=scale, stride=scale)
        noise_ratio = (in_s + eps) / (pr_s + eps)
        losses.append(gamma_speckle_nll(noise_ratio, looks=looks, eps=eps))
    return torch.stack(losses).mean()


def enl_consistency_loss(
    pred_01: torch.Tensor,
    patch_size: int,
    expected_enl: float,
    homogeneity_cv_max: float,
    eps: float,
) -> torch.Tensor:
    mean = torch.nn.functional.avg_pool2d(pred_01, kernel_size=patch_size, stride=1, padding=patch_size // 2)
    mean_sq = torch.nn.functional.avg_pool2d(pred_01 * pred_01, kernel_size=patch_size, stride=1, padding=patch_size // 2)
    var = (mean_sq - mean * mean).clamp(min=0.0)
    std = torch.sqrt(var + eps)
    cv = std / (mean + eps)
    mask = (cv < homogeneity_cv_max).float()
    enl = (mean * mean) / (var + eps)
    loss_map = torch.abs(torch.log(enl + eps) - math.log(max(expected_enl, eps)))
    denom = mask.sum().clamp(min=1.0)
    return (loss_map * mask).sum() / denom


class EMA:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.ema_model = copy.deepcopy(model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        ema_state = self.ema_model.state_dict()
        model_state = model.state_dict()
        for key, ema_value in ema_state.items():
            model_value = model_state[key]
            if not torch.is_floating_point(ema_value):
                ema_value.copy_(model_value)
            else:
                ema_value.mul_(self.decay).add_(model_value.detach(), alpha=1.0 - self.decay)

    def state_dict(self) -> dict:
        return self.ema_model.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        self.ema_model.load_state_dict(state_dict)


def build_scheduler(optimizer, total_steps: int, warmup_steps: int, min_lr_ratio: float) -> LambdaLR:
    warmup_steps = max(0, warmup_steps)
    total_steps = max(total_steps, 1)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)


def train() -> None:
    args = parse_args()
    config = Config.from_yaml(args.config).raw
    set_seed(config["seed"])

    runtime_cfg = config.get("runtime", {})
    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["train"]
    phys_cfg = train_cfg.get("physics_loss", {})
    stability_cfg = train_cfg.get("stability", {})

    device_name = args.device or runtime_cfg.get("device") or config.get("device", "cuda")
    device = resolve_device(device_name)
    amp_enabled = bool(runtime_cfg.get("amp", device.type == "cuda"))
    configure_cuda_runtime(runtime_cfg=runtime_cfg, device=device)

    grad_clip_norm = float(train_cfg.get("grad_clip_norm", 1.0))
    accum_steps = int(stability_cfg.get("grad_accum_steps", 1))
    ema_decay = float(stability_cfg.get("ema_decay", 0.999))
    use_ema = bool(stability_cfg.get("use_ema", True))

    lambda_charb = float(train_cfg.get("lambda_charbonnier", 1.0))
    lambda_grad = float(train_cfg.get("lambda_grad", 0.2))
    lambda_frc = float(train_cfg.get("lambda_frc", 0.5))
    lambda_smooth = float(train_cfg.get("lambda_smooth", 0.03))
    lambda_log_mean = float(train_cfg.get("lambda_log_noise_mean", 0.05))
    lambda_log_tv = float(train_cfg.get("lambda_log_noise_tv", 0.03))
    lambda_gamma_nll = float(train_cfg.get("lambda_gamma_nll", 0.0))
    lambda_enl = float(train_cfg.get("lambda_enl", 0.0))

    speckle_eps = float(phys_cfg.get("eps", 1e-3))
    frc_rings = int(phys_cfg.get("frc_rings", 32))
    gamma_looks = float(phys_cfg.get("gamma_looks", 1.0))
    gamma_scales = tuple(int(s) for s in phys_cfg.get("gamma_scales", [1, 2, 4]))
    expected_enl = float(phys_cfg.get("expected_enl", max(gamma_looks, 1.0)))
    enl_patch_size = int(phys_cfg.get("enl_patch_size", 7))
    enl_homogeneity_cv_max = float(phys_cfg.get("enl_homogeneity_cv_max", 0.12))
    use_weighted_frc = bool(phys_cfg.get("use_weighted_frc", True))
    frc_low_freq_weight = float(phys_cfg.get("frc_low_freq_weight", 1.0))
    frc_high_freq_weight = float(phys_cfg.get("frc_high_freq_weight", 1.2))

    train_loader = create_paired_train_dataloader(
        root_dir=data_cfg["train_dir"],
        image_size=data_cfg["image_size"],
        channels=data_cfg["channels"],
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
        pin_memory=device.type == "cuda",
        persistent_workers=bool(data_cfg.get("persistent_workers", device.type == "cuda")),
        equalize_speckle=bool(data_cfg.get("equalize_speckle", True)),
    )
    val_loader = create_validation_dataloader(
        root_dir=data_cfg["val_dir"],
        image_size=data_cfg["image_size"],
        channels=data_cfg["channels"],
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
        pin_memory=device.type == "cuda",
        persistent_workers=bool(data_cfg.get("persistent_workers", device.type == "cuda")),
        equalize_speckle=bool(data_cfg.get("equalize_speckle", True)),
    )

    model = SwinUNetDenoiser(
        in_channels=data_cfg["channels"],
        out_channels=data_cfg["channels"],
        embed_dim=int(model_cfg.get("embed_dim", 64)),
        depths=tuple(model_cfg.get("depths", [2, 2, 2, 2])),
        num_heads=tuple(model_cfg.get("num_heads", [2, 4, 8, 16])),
        window_size=int(model_cfg.get("window_size", 8)),
        mlp_ratio=float(model_cfg.get("mlp_ratio", 4.0)),
        drop_rate=float(model_cfg.get("drop_rate", 0.0)),
        attn_drop_rate=float(model_cfg.get("attn_drop_rate", 0.0)),
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        betas=tuple(train_cfg["betas"]),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and device.type == "cuda")

    steps_per_epoch = max(len(train_loader), 1)
    total_steps = train_cfg["epochs"] * math.ceil(steps_per_epoch / max(accum_steps, 1))
    scheduler = build_scheduler(
        optimizer=optimizer,
        total_steps=total_steps,
        warmup_steps=int(stability_cfg.get("warmup_steps", max(total_steps // 20, 1))),
        min_lr_ratio=float(stability_cfg.get("min_lr_ratio", 0.05)),
    )

    ema = EMA(model, decay=ema_decay) if use_ema else None

    run_dir = ensure_dir(Path(config["output_dir"]) / config["experiment_name"])
    ckpt_dir = ensure_dir(run_dir / "checkpoints")
    preview_dir = ensure_dir(run_dir / "previews")
    metrics_csv = run_dir / "metrics.csv"
    metrics_jsonl = run_dir / "metrics.jsonl"

    best_val_l1 = float("inf")
    global_step = 0

    for epoch in range(1, train_cfg["epochs"] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        epoch_loss = 0.0
        epoch_charb = 0.0
        epoch_grad = 0.0
        epoch_frc = 0.0
        epoch_smooth = 0.0
        epoch_log_mean = 0.0
        epoch_log_tv = 0.0
        epoch_gamma_nll = 0.0
        epoch_enl = 0.0
        num_batches = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{train_cfg['epochs']}", leave=False)
        for step_idx, batch in enumerate(progress, start=1):
            real_a = batch["domain_a"].to(device, non_blocking=device.type == "cuda")
            real_b = batch["domain_b"].to(device, non_blocking=device.type == "cuda")

            with autocast_context(device, amp_enabled):
                pred = model(real_a)

                pred_01 = to_unit_range(pred)
                input_01 = to_unit_range(real_a)
                target_01 = to_unit_range(real_b)

                log_noise = torch.log((input_01 + speckle_eps) / (pred_01 + speckle_eps))
                noise_ratio = (input_01 + speckle_eps) / (pred_01 + speckle_eps)

                loss_charb = charbonnier_loss(pred, real_b) * lambda_charb
                loss_grad = torch.abs(gradient_map(pred_01) - gradient_map(target_01)).mean() * lambda_grad
                frc_base = (
                    weighted_frc_loss(
                        pred_01,
                        target_01,
                        num_rings=frc_rings,
                        low_freq_weight=frc_low_freq_weight,
                        high_freq_weight=frc_high_freq_weight,
                    )
                    if use_weighted_frc
                    else frc_loss(pred_01, target_01, num_rings=frc_rings)
                )
                loss_frc_term = frc_base * lambda_frc
                loss_smooth = total_variation_loss(pred_01) * lambda_smooth
                loss_log_mean = (log_noise.mean(dim=(1, 2, 3)) ** 2).mean() * lambda_log_mean
                loss_log_tv = total_variation_loss(log_noise) * lambda_log_tv
                loss_gamma_nll = (
                    multiscale_gamma_speckle_nll(
                        input_01=input_01,
                        pred_01=pred_01,
                        looks=gamma_looks,
                        eps=speckle_eps,
                        scales=gamma_scales,
                    )
                    * lambda_gamma_nll
                )
                loss_enl = (
                    enl_consistency_loss(
                        pred_01=pred_01,
                        patch_size=enl_patch_size,
                        expected_enl=expected_enl,
                        homogeneity_cv_max=enl_homogeneity_cv_max,
                        eps=speckle_eps,
                    )
                    * lambda_enl
                )

                loss = (
                    loss_charb
                    + loss_grad
                    + loss_frc_term
                    + loss_smooth
                    + loss_log_mean
                    + loss_log_tv
                    + loss_gamma_nll
                    + loss_enl
                )

            scaler.scale(loss / max(accum_steps, 1)).backward()
            do_step = (step_idx % max(accum_steps, 1) == 0) or (step_idx == len(train_loader))

            if do_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                prev_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer_step_ran = scaler.get_scale() >= prev_scale
                optimizer.zero_grad(set_to_none=True)
                if optimizer_step_ran:
                    scheduler.step()
                    global_step += 1
                    if ema is not None:
                        ema.update(model)


            epoch_loss += loss.item()
            epoch_charb += loss_charb.item()
            epoch_grad += loss_grad.item()
            epoch_frc += loss_frc_term.item()
            epoch_smooth += loss_smooth.item()
            epoch_log_mean += loss_log_mean.item()
            epoch_log_tv += loss_log_tv.item()
            epoch_gamma_nll += loss_gamma_nll.item()
            epoch_enl += loss_enl.item()
            num_batches += 1

            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                frc=f"{loss_frc_term.item():.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

        eval_model = ema.ema_model if ema is not None else model
        val_metrics = validate(eval_model, val_loader, device, preview_dir, epoch, frc_rings)

        epoch_metrics = {
            "epoch": epoch,
            "timestamp_utc": utc_timestamp(),
            "global_step": global_step,
            "train_loss": epoch_loss / max(num_batches, 1),
            "train_charbonnier_loss": epoch_charb / max(num_batches, 1),
            "train_grad_loss": epoch_grad / max(num_batches, 1),
            "train_frc_loss": epoch_frc / max(num_batches, 1),
            "train_smooth_loss": epoch_smooth / max(num_batches, 1),
            "train_log_noise_mean_loss": epoch_log_mean / max(num_batches, 1),
            "train_log_noise_tv_loss": epoch_log_tv / max(num_batches, 1),
            "train_gamma_nll_loss": epoch_gamma_nll / max(num_batches, 1),
            "train_enl_loss": epoch_enl / max(num_batches, 1),
            "lr": optimizer.param_groups[0]["lr"],
            **val_metrics,
        }
        append_metrics(metrics_csv, metrics_jsonl, epoch_metrics)

        checkpoint = {
            "epoch": epoch,
            "model_name": "swin_unet_physics_denoiser",
            "model_state_dict": model.state_dict(),
            "ema_state_dict": ema.state_dict() if ema is not None else None,
            "config": config,
            "val_l1_loss": val_metrics["val_l1_loss"],
        }
        torch.save(checkpoint, ckpt_dir / "last.pt")
        if epoch % train_cfg["save_every"] == 0:
            torch.save(checkpoint, ckpt_dir / f"epoch_{epoch:03d}.pt")
        if val_metrics["val_l1_loss"] < best_val_l1:
            best_val_l1 = val_metrics["val_l1_loss"]
            torch.save(checkpoint, run_dir / "best.pt")

        print(
            f"Epoch {epoch}: "
            f"train_loss={epoch_metrics['train_loss']:.6f}, "
            f"val_l1={epoch_metrics['val_l1_loss']:.6f}, "
            f"val_frc={epoch_metrics['val_frc_score']:.4f}, "
            f"val_psnr={epoch_metrics['val_psnr']:.4f}, "
            f"val_ssim={epoch_metrics['val_ssim']:.4f}"
        )


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader,
    device: torch.device,
    preview_dir: Path,
    epoch: int,
    frc_rings: int,
) -> dict[str, float]:
    model.eval()
    criterion_l1 = nn.L1Loss()
    l1_losses = []
    psnr_scores = []
    ssim_scores = []
    frc_scores = []
    preview_saved = False

    for batch in val_loader:
        real_a = batch["domain_a"].to(device, non_blocking=device.type == "cuda")
        real_b = batch["domain_b"].to(device, non_blocking=device.type == "cuda")
        pred = model(real_a)
        pred_01 = to_unit_range(pred)
        real_b_01 = to_unit_range(real_b)

        l1_losses.append(criterion_l1(pred, real_b).item())
        psnr_scores.append(compute_psnr(pred, real_b))
        ssim_scores.append(compute_ssim(pred, real_b))
        frc_scores.append((1.0 - frc_loss(pred_01, real_b_01, num_rings=frc_rings)).item())

        if not preview_saved:
            save_preview(real_a.cpu(), pred.cpu(), real_b.cpu(), preview_dir / f"epoch_{epoch:03d}.png")
            preview_saved = True

    return {
        "val_l1_loss": sum(l1_losses) / max(len(l1_losses), 1),
        "val_psnr": sum(psnr_scores) / max(len(psnr_scores), 1),
        "val_ssim": sum(ssim_scores) / max(len(ssim_scores), 1),
        "val_frc_score": sum(frc_scores) / max(len(frc_scores), 1),
    }


if __name__ == "__main__":
    train()
