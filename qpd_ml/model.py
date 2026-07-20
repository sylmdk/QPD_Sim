from __future__ import annotations

from collections.abc import Callable
from typing import Any

import lightning as L
import torch
from torch import nn
from torch.nn import functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from .color import (
    batch_to_srgb,
    make_comparison_grid,
    packed_qpd_to_camera_rgb,
    qpd_input_to_srgb,
)


def _activation(name: str) -> nn.Module:
    activations: dict[str, Callable[[], nn.Module]] = {
        "relu": lambda: nn.ReLU(inplace=True),
        "leaky_relu": lambda: nn.LeakyReLU(0.1, inplace=True),
        "silu": lambda: nn.SiLU(inplace=True),
        "gelu": nn.GELU,
    }
    try:
        return activations[name.lower()]()
    except KeyError as exc:
        raise ValueError(f"Unsupported activation: {name}") from exc


def _normalization(name: str, channels: int, groups: int) -> nn.Module:
    name = name.lower()
    if name == "group":
        valid_groups = min(groups, channels)
        while channels % valid_groups:
            valid_groups -= 1
        return nn.GroupNorm(valid_groups, channels)
    if name == "batch":
        return nn.BatchNorm2d(channels)
    if name in ("none", "identity"):
        return nn.Identity()
    raise ValueError(f"Unsupported normalization: {name}")


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        normalization: str,
        groups: int,
        activation: str,
        dropout: float,
        residual: bool,
    ) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            _normalization(normalization, out_channels, groups),
            _activation(activation),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            _normalization(normalization, out_channels, groups),
        )
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
            if residual and in_channels != out_channels else nn.Identity()
        )
        self.residual = residual
        self.out_activation = _activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.body(x)
        if self.residual:
            y = y + self.skip(x)
        return self.out_activation(y)


class BlockStack(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, count: int, **kwargs: Any) -> None:
        blocks = [ResidualBlock(in_channels, out_channels, **kwargs)]
        blocks.extend(ResidualBlock(out_channels, out_channels, **kwargs) for _ in range(count - 1))
        super().__init__(*blocks)


class ConfigurableUNet(nn.Module):
    """Residual U-Net on packed Bayer planes with full-resolution RGB output."""
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        base_channels: int = 32,
        channel_multipliers: list[int] | tuple[int, ...] = (1, 2, 4, 8),
        blocks_per_stage: int = 2,
        normalization: str = "group",
        norm_groups: int = 8,
        activation: str = "silu",
        dropout: float = 0.0,
        residual_blocks: bool = True,
        upsample_mode: str = "bilinear",
        output_activation: str = "sigmoid",
    ) -> None:
        super().__init__()
        if len(channel_multipliers) < 2:
            raise ValueError("channel_multipliers must contain at least two stages")
        if blocks_per_stage < 1:
            raise ValueError("blocks_per_stage must be >= 1")
        if upsample_mode not in ("bilinear", "transpose"):
            raise ValueError("upsample_mode must be 'bilinear' or 'transpose'")
        channels = [base_channels * int(m) for m in channel_multipliers]
        block_kw = dict(normalization=normalization, groups=norm_groups,
                        activation=activation, dropout=dropout, residual=residual_blocks)
        self.stem = nn.Conv2d(in_channels, channels[0], 3, padding=1)
        self.encoders = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for index, channel in enumerate(channels):
            self.encoders.append(BlockStack(channel, channel, blocks_per_stage, **block_kw))
            if index < len(channels) - 1:
                self.downsamples.append(nn.Conv2d(channel, channels[index + 1], 3, stride=2, padding=1))
        self.bottleneck = BlockStack(channels[-1], channels[-1], blocks_per_stage, **block_kw)
        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for index in range(len(channels) - 1, 0, -1):
            if upsample_mode == "transpose":
                up = nn.ConvTranspose2d(channels[index], channels[index - 1], 2, stride=2)
            else:
                up = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    nn.Conv2d(channels[index], channels[index - 1], 3, padding=1),
                )
            self.upsamples.append(up)
            self.decoders.append(BlockStack(
                channels[index - 1] * 2, channels[index - 1], blocks_per_stage, **block_kw
            ))
        head: list[nn.Module] = [nn.Conv2d(channels[0], out_channels * 4, 3, padding=1), nn.PixelShuffle(2)]
        if output_activation == "sigmoid":
            head.append(nn.Sigmoid())
        elif output_activation not in ("none", "identity"):
            raise ValueError("output_activation must be 'sigmoid' or 'none'")
        self.head = nn.Sequential(*head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        skips = []
        for index, encoder in enumerate(self.encoders):
            x = encoder(x)
            skips.append(x)
            if index < len(self.downsamples):
                x = self.downsamples[index](x)
        x = self.bottleneck(x)
        for upsample, decoder, skip in zip(self.upsamples, self.decoders, reversed(skips[:-1])):
            x = upsample(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = decoder(torch.cat((x, skip), dim=1))
        return self.head(x)


class QPDLightningModule(L.LightningModule):
    def __init__(
        self,
        model: dict[str, Any],
        optimizer: dict[str, Any],
        loss: dict[str, Any],
        metrics: dict[str, Any],
        visualization: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.net = ConfigurableUNet(**model)
        self.loss_config = loss
        self.metrics_config = metrics
        if metrics.get("domain", "srgb") not in ("srgb", "camera_linear"):
            raise ValueError("metrics.domain must be 'srgb' or 'camera_linear'")
        self.visualization_config = visualization or {}
        data_range = float(metrics.get("data_range", 1.0))
        self.val_ssim = StructuralSimilarityIndexMeasure(data_range=data_range)
        self.test_ssim = StructuralSimilarityIndexMeasure(data_range=data_range)
        lpips_cfg = metrics.get("lpips", {})
        self.lpips_enabled = bool(lpips_cfg.get("enabled", True))
        if self.lpips_enabled:
            self.lpips_metric = LearnedPerceptualImagePatchSimilarity(
                net_type=lpips_cfg.get("net_type", "alex"), normalize=True
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def _metric_images(
        self, prediction: torch.Tensor, target: torch.Tensor, batch: dict
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.metrics_config.get("domain", "srgb") == "srgb":
            return batch_to_srgb(prediction, batch), batch_to_srgb(target, batch)
        return prediction.clamp(0.0, 1.0), target.clamp(0.0, 1.0)

    def _step(self, batch: dict, stage: str, batch_idx: int) -> torch.Tensor:
        prediction = self(batch["input"])
        target = batch["target"]
        l1 = F.l1_loss(prediction, target)
        mse = F.mse_loss(prediction, target)
        loss = float(self.loss_config.get("l1_weight", 1.0)) * l1
        if self.loss_config.get("mse_weight", 0.0):
            loss = loss + float(self.loss_config["mse_weight"]) * mse
        metric_pred, metric_target = self._metric_images(prediction, target, batch)
        metric_mse = F.mse_loss(metric_pred, metric_target)
        psnr = -10.0 * torch.log10(metric_mse.clamp_min(1e-10))
        values = {"loss": loss, "l1": l1, "psnr": psnr}
        if stage != "train":
            ssim_metric = self.val_ssim if stage == "val" else self.test_ssim
            values["ssim"] = ssim_metric(metric_pred, metric_target)
            if self.lpips_enabled:
                values["lpips"] = self.lpips_metric(metric_pred, metric_target)
        for name, value in values.items():
            self.log(f"{stage}_{name}", value, on_step=stage == "train", on_epoch=True,
                     prog_bar=name in ("loss", "psnr", "ssim"), batch_size=target.shape[0],
                     sync_dist=self.trainer.world_size > 1)
        if stage == "val" and batch_idx == 0:
            self._log_validation_images(batch, prediction, target)
        return loss

    def _log_validation_images(self, batch: dict, prediction: torch.Tensor, target: torch.Tensor) -> None:
        cfg = self.visualization_config
        interval = max(1, int(cfg.get("every_n_epochs", 1)))
        if (
            not cfg.get("enabled", True)
            or self.current_epoch % interval
            or self.logger is None
            or not self.trainer.is_global_zero
        ):
            return
        experiment = getattr(self.logger, "experiment", None)
        if experiment is None or not hasattr(experiment, "add_images"):
            return
        with torch.no_grad():
            domain = cfg.get("domain", "srgb")
            if domain == "srgb":
                display_prediction = batch_to_srgb(prediction, batch)
                display_target = batch_to_srgb(target, batch)
                display_input = qpd_input_to_srgb(batch["input"], batch, target.shape[-2:])
            elif domain == "camera_linear":
                display_prediction = prediction.clamp(0.0, 1.0)
                display_target = target.clamp(0.0, 1.0)
                display_input = packed_qpd_to_camera_rgb(batch["input"], target.shape[-2:])
            else:
                raise ValueError("visualization.domain must be 'srgb' or 'camera_linear'")
            grid = make_comparison_grid(
                display_input, display_prediction, display_target, int(cfg.get("max_images", 4))
            )
        experiment.add_images("validation/input_pred_target_error", grid, self.global_step)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train", batch_idx)

    def on_validation_epoch_start(self) -> None:
        self.val_ssim.reset()
        if self.lpips_enabled:
            self.lpips_metric.reset()

    def on_test_epoch_start(self) -> None:
        self.test_ssim.reset()
        if self.lpips_enabled:
            self.lpips_metric.reset()

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        self._step(batch, "val", batch_idx)

    def test_step(self, batch: dict, batch_idx: int) -> None:
        self._step(batch, "test", batch_idx)

    def configure_optimizers(self):
        cfg = self.hparams.optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=float(cfg.get("learning_rate", 2e-4)),
            weight_decay=float(cfg.get("weight_decay", 1e-4)),
            betas=tuple(cfg.get("betas", (0.9, 0.999))),
        )
        scheduler_name = cfg.get("scheduler", "cosine").lower()
        if scheduler_name == "none":
            return optimizer
        if scheduler_name != "cosine":
            raise ValueError(f"Unsupported scheduler: {scheduler_name}")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, self.trainer.max_epochs),
            eta_min=float(cfg.get("min_learning_rate", 1e-6)),
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
