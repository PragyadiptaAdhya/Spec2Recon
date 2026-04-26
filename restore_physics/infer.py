from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.utils import save_image

from models import SwinUNetDenoiser


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Restore-Physics inference from speckle to restored image")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--input", type=str, required=True, help="Input speckle image")
    parser.add_argument("--output", type=str, required=True, help="Output restored image")
    parser.add_argument("--use-ema", action="store_true", help="Use EMA weights when available")
    return parser.parse_args()


def resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint["config"]
    data_cfg = config["data"]
    model_cfg = config["model"]

    if checkpoint.get("model_name") != "swin_unet_physics_denoiser":
        raise ValueError("Unsupported checkpoint type. Expected `swin_unet_physics_denoiser`.")

    device = resolve_device()
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

    state_dict = checkpoint["model_state_dict"]
    if args.use_ema and checkpoint.get("ema_state_dict") is not None:
        state_dict = checkpoint["ema_state_dict"]
    model.load_state_dict(state_dict)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1)
            if data_cfg["channels"] == 1
            else transforms.Lambda(lambda x: x),
            transforms.Resize((data_cfg["image_size"], data_cfg["image_size"])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * data_cfg["channels"], std=[0.5] * data_cfg["channels"]),
        ]
    )

    image = Image.open(args.input).convert("RGB" if data_cfg["channels"] == 3 else "L")
    if data_cfg.get("equalize_speckle", True):
        image = ImageOps.equalize(image)
    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(x)

    pred = (pred.cpu() + 1.0) / 2.0
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(pred, output_path)


if __name__ == "__main__":
    main()
