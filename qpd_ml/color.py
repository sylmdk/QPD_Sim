from __future__ import annotations

import torch
from torch.nn import functional as F


def linear_to_srgb(linear_rgb: torch.Tensor) -> torch.Tensor:
    """Differentiable IEC 61966-2-1 transfer function for NCHW tensors."""
    x = linear_rgb.clamp(0.0, 1.0)
    return torch.where(
        x <= 0.0031308,
        x * 12.92,
        1.055 * torch.pow(x.clamp_min(0.0031308), 1.0 / 2.4) - 0.055,
    )


def camera_rgb_to_linear_srgb(
    camera_rgb: torch.Tensor,
    wb_gains: torch.Tensor,
    ccm_srgb_from_cam: torch.Tensor,
) -> torch.Tensor:
    """Apply per-image white balance and camera-to-sRGB matrix in linear space."""
    if camera_rgb.ndim != 4 or camera_rgb.shape[1] != 3:
        raise ValueError("camera_rgb must have shape [B, 3, H, W]")
    wb = wb_gains.to(camera_rgb).view(-1, 3, 1, 1)
    ccm = ccm_srgb_from_cam.to(camera_rgb).view(-1, 3, 3)
    balanced = camera_rgb * wb
    return torch.einsum("bij,bjhw->bihw", ccm, balanced)


def camera_rgb_to_srgb(
    camera_rgb: torch.Tensor,
    wb_gains: torch.Tensor,
    ccm_srgb_from_cam: torch.Tensor,
) -> torch.Tensor:
    return linear_to_srgb(camera_rgb_to_linear_srgb(camera_rgb, wb_gains, ccm_srgb_from_cam))


def packed_qpd_to_camera_rgb(packed: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
    """Create a display preview from packed [R, Gr, Gb, B] QPD planes."""
    if packed.ndim != 4 or packed.shape[1] != 4:
        raise ValueError("packed QPD must have shape [B, 4, H, W]")
    camera_rgb = torch.stack(
        (packed[:, 0], 0.5 * (packed[:, 1] + packed[:, 2]), packed[:, 3]), dim=1
    )
    return F.interpolate(camera_rgb, size=output_size, mode="bilinear", align_corners=False)


def batch_to_srgb(
    camera_rgb: torch.Tensor,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    return camera_rgb_to_srgb(camera_rgb, batch["wb_gains"], batch["ccm"])


def qpd_input_to_srgb(
    packed: torch.Tensor,
    batch: dict[str, torch.Tensor],
    output_size: tuple[int, int],
) -> torch.Tensor:
    camera_rgb = packed_qpd_to_camera_rgb(packed, output_size)
    return batch_to_srgb(camera_rgb, batch)


def make_comparison_grid(
    input_srgb: torch.Tensor,
    prediction_srgb: torch.Tensor,
    target_srgb: torch.Tensor,
    max_images: int = 4,
) -> torch.Tensor:
    """Return rows of input | prediction | target | amplified absolute error."""
    count = min(max_images, input_srgb.shape[0])
    rows = []
    for index in range(count):
        error = (prediction_srgb[index] - target_srgb[index]).abs().mul(4.0).clamp(0.0, 1.0)
        rows.append(torch.cat(
            (input_srgb[index], prediction_srgb[index], target_srgb[index], error), dim=2
        ))
    return torch.stack(rows).detach().clamp(0.0, 1.0)
