from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _build_transform(image_size: int, channels: int) -> transforms.Compose:
    mode_transform = transforms.Grayscale(num_output_channels=1) if channels == 1 else None
    transform_steps = []
    if mode_transform is not None:
        transform_steps.append(mode_transform)
    transform_steps.extend(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * channels, std=[0.5] * channels),
        ]
    )
    return transforms.Compose(transform_steps)


class PairedSpeckleDataset(Dataset):
    def __init__(
        self,
        root_dir: str | Path,
        image_size: int,
        channels: int = 1,
        equalize_speckle: bool = True,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.speckle_dir = self.root_dir / "speckle"
        self.gt_dir = self.root_dir / "groundtruth"
        self.channels = channels
        self.equalize_speckle = equalize_speckle

        speckle_files = sorted(self._iter_image_files(self.speckle_dir))
        self.pairs: list[tuple[Path, Path]] = []
        for speckle_path in speckle_files:
            relative_path = speckle_path.relative_to(self.speckle_dir)
            gt_path = self.gt_dir / relative_path
            if gt_path.exists():
                self.pairs.append((speckle_path, gt_path))

        if not self.pairs:
            raise RuntimeError(f"No paired validation images found in {self.root_dir}")

        self.transform = _build_transform(image_size=image_size, channels=channels)

    @staticmethod
    def _iter_image_files(root_dir: Path) -> Iterable[Path]:
        for path in root_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                yield path

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, object]:
        speckle_path, gt_path = self.pairs[index]
        speckle_img = Image.open(speckle_path).convert("RGB" if self.channels == 3 else "L")
        gt_img = Image.open(gt_path).convert("RGB" if self.channels == 3 else "L")
        if self.equalize_speckle:
            speckle_img = ImageOps.equalize(speckle_img)
        return {
            "domain_a": self.transform(speckle_img),
            "domain_b": self.transform(gt_img),
            "name": speckle_path.name,
        }


def create_validation_dataloader(
    root_dir: str | Path,
    image_size: int,
    channels: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool = True,
    persistent_workers: bool = False,
    equalize_speckle: bool = True,
) -> DataLoader:
    dataset = PairedSpeckleDataset(
        root_dir=root_dir,
        image_size=image_size,
        channels=channels,
        equalize_speckle=equalize_speckle,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
    )


def create_paired_train_dataloader(
    root_dir: str | Path,
    image_size: int,
    channels: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool = True,
    persistent_workers: bool = False,
    equalize_speckle: bool = True,
) -> DataLoader:
    dataset = PairedSpeckleDataset(
        root_dir=root_dir,
        image_size=image_size,
        channels=channels,
        equalize_speckle=equalize_speckle,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
    )
