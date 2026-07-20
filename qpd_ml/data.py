from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def pack_quad_bayer(raw: np.ndarray) -> np.ndarray:
    """Pack a 2x2 RGGB mosaic to four half-resolution planes."""
    return np.stack(
        (raw[0::2, 0::2], raw[0::2, 1::2], raw[1::2, 0::2], raw[1::2, 1::2]),
        axis=0,
    )


def normalize_qpd_raw(raw: np.ndarray, metadata: dict) -> np.ndarray:
    """Normalize QPD code values using the output black/white levels."""
    levels = metadata["qpd_output_levels"]
    denominator = float(levels["white_level"]) - float(levels["black_level"])
    if denominator <= 0:
        raise ValueError("QPD white level must be greater than black level")
    return np.clip(
        (np.asarray(raw, dtype=np.float32) - float(levels["black_level"])) / denominator,
        0.0,
        1.0,
    )


class QPDPatchDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        patch_size: int = 256,
        patches_per_image: int = 8,
        random_crop: bool | None = None,
        augment: bool = False,
        augment_rotate_180: bool = True,
        seed: int = 2026,
    ) -> None:
        self.root = Path(root).resolve()
        self.split = split
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.random_crop = split == "train" if random_crop is None else random_crop
        self.augment = augment
        self.augment_rotate_180 = augment_rotate_180
        self.seed = seed
        if patch_size <= 0 or patch_size % 16:
            raise ValueError("patch_size must be positive and divisible by 16")
        csv_path = self.root / f"{split}.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"Missing split manifest: {csv_path}")
        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            self.rows = list(csv.DictReader(f))
        if not self.rows:
            raise ValueError(f"No samples in {csv_path}")

    def __len__(self) -> int:
        return len(self.rows) * self.patches_per_image

    def _origin(self, h: int, w: int, index: int) -> tuple[int, int]:
        p = self.patch_size
        if h < p or w < p:
            raise ValueError(f"Image {h}x{w} is smaller than patch_size={p}")
        if self.random_crop:
            top = random.randrange(0, h - p + 1)
            left = random.randrange(0, w - p + 1)
        else:
            # Repeatable, spatially distributed evaluation crops.
            local = index % self.patches_per_image
            rng = random.Random(self.seed + local * 104729)
            top = rng.randrange(0, h - p + 1)
            left = rng.randrange(0, w - p + 1)
        return top - top % 2, left - left % 2

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.rows[index // self.patches_per_image]
        raw_path = self.root / Path(row["input_qpd_raw"])
        target_path = self.root / Path(row["target_clean_energy"])
        meta_path = self.root / Path(row["metadata"])
        raw = np.load(raw_path, mmap_mode="r")
        target = np.load(target_path, mmap_mode="r")
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        top, left = self._origin(raw.shape[0], raw.shape[1], index)
        p = self.patch_size
        raw_patch = np.asarray(raw[top : top + p, left : left + p], dtype=np.float32)
        target_patch = np.asarray(target[top : top + p, left : left + p], dtype=np.float32)
        raw_patch = normalize_qpd_raw(raw_patch, meta)
        x = pack_quad_bayer(raw_patch).copy()
        y = np.moveaxis(target_patch, -1, 0).copy()
        if self.augment and self.augment_rotate_180:
            # 180-degree rotation and paired flips preserve CFA plane semantics.
            if random.random() < 0.5:
                x, y = x[:, ::-1, ::-1].copy(), y[:, ::-1, ::-1].copy()
        return {
            "input": torch.from_numpy(x),
            "target": torch.from_numpy(y),
            "sample_id": row["sample_id"],
            "wb_gains": torch.tensor(meta["isp_params"]["wb_gains"], dtype=torch.float32),
            "ccm": torch.tensor(meta["isp_params"]["ccm_srgb_from_cam"], dtype=torch.float32),
        }


class QPDDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str | Path,
        patch_size: int = 256,
        batch_size: int = 2,
        train_patches_per_image: int = 8,
        eval_patches_per_image: int = 4,
        num_workers: int = 0,
        seed: int = 2026,
        augment_rotate_180: bool = True,
        data_split: str = "train",
    ) -> None:
        super().__init__()
        # Keep checkpoints compatible with PyTorch's safe weights-only loader.
        data_dir = str(Path(data_dir))
        self.save_hyperparameters()

    def setup(self, stage: str | None = None) -> None:
        kw = dict(root=self.hparams.data_dir, patch_size=self.hparams.patch_size)
        if stage in (None, "fit"):
            self.train_set = QPDPatchDataset(
                **kw, split=self.hparams.data_split,
                patches_per_image=self.hparams.train_patches_per_image,
                augment=True, seed=self.hparams.seed,
                augment_rotate_180=self.hparams.augment_rotate_180,
            )
            self.val_set = QPDPatchDataset(
                **kw, split="val",
                patches_per_image=self.hparams.eval_patches_per_image,
                random_crop=False, seed=self.hparams.seed,
            )
        if stage in (None, "test"):
            self.test_set = QPDPatchDataset(
                **kw, split=self.hparams.data_split,
                patches_per_image=self.hparams.eval_patches_per_image,
                random_crop=False, seed=self.hparams.seed,
            )

    def _loader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset, batch_size=self.hparams.batch_size, shuffle=shuffle,
            num_workers=self.hparams.num_workers, pin_memory=torch.cuda.is_available(),
            persistent_workers=self.hparams.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_set, True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_set, False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_set, False)
