from __future__ import annotations

import csv
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import save_image


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def append_metrics(csv_path: Path, jsonl_path: Path, metrics: dict[str, float | int | str]) -> None:
    fieldnames = list(metrics.keys())
    rewrite_csv = False
    existing_rows: list[dict[str, str]] = []
    if csv_path.exists():
        with csv_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != fieldnames:
                rewrite_csv = True
                existing_rows = list(reader)

    mode = "w" if rewrite_csv or not csv_path.exists() else "a"
    with csv_path.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if mode == "w":
            writer.writeheader()
            for row in existing_rows:
                writer.writerow({name: row.get(name, "") for name in fieldnames})
        writer.writerow(metrics)
        handle.flush()

    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(metrics) + "\n")
        handle.flush()


def compute_psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    mse = F.mse_loss(pred, target).item()
    if mse <= eps:
        return 100.0
    return float(10.0 * torch.log10(torch.tensor(4.0 / mse)).item())


def compute_ssim(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    pred = (pred + 1.0) / 2.0
    target = (target + 1.0) / 2.0
    mu_x = pred.mean(dim=(-2, -1), keepdim=True)
    mu_y = target.mean(dim=(-2, -1), keepdim=True)
    sigma_x = ((pred - mu_x) ** 2).mean(dim=(-2, -1), keepdim=True)
    sigma_y = ((target - mu_y) ** 2).mean(dim=(-2, -1), keepdim=True)
    sigma_xy = ((pred - mu_x) * (target - mu_y)).mean(dim=(-2, -1), keepdim=True)
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x.pow(2) + mu_y.pow(2) + c1) * (sigma_x + sigma_y + c2)
    return float((numerator / (denominator + eps)).mean().item())


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_preview(
    real_a: torch.Tensor,
    fake_b: torch.Tensor,
    real_b: torch.Tensor,
    output_path: str | Path,
) -> None:
    real_a = (real_a + 1.0) / 2.0
    fake_b = (fake_b + 1.0) / 2.0
    real_b = (real_b + 1.0) / 2.0
    grid = torch.cat([real_a, fake_b, real_b], dim=0)
    save_image(grid, output_path, nrow=real_a.size(0))
