import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def linear_to_srgb(linear_rgb):
    """Apply the IEC 61966-2-1 sRGB transfer function."""
    linear_rgb = np.clip(np.asarray(linear_rgb, dtype=np.float32), 0.0, 1.0)
    return np.where(
        linear_rgb <= 0.0031308,
        linear_rgb * 12.92,
        1.055 * np.power(linear_rgb, 1.0 / 2.4) - 0.055,
    ).astype(np.float32)


def load_metadata(path):
    metadata_path = Path(path).resolve()
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata does not exist: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    if not isinstance(metadata, dict):
        raise ValueError(f"Metadata root must be a JSON object: {metadata_path}")
    return metadata_path, metadata


def load_linear_srgb(path, metadata):
    linear_path = Path(path).expanduser().resolve()
    if not linear_path.is_file():
        raise FileNotFoundError(f"Linear sRGB NPY does not exist: {linear_path}")
    linear_srgb = np.load(linear_path, allow_pickle=False)
    if linear_srgb.ndim != 3 or linear_srgb.shape[2] != 3:
        raise ValueError(
            f"isp_linear_srgb must have shape (H,W,3), got {linear_srgb.shape}"
        )
    if not np.issubdtype(linear_srgb.dtype, np.number):
        raise TypeError(f"isp_linear_srgb must be numeric, got {linear_srgb.dtype}")
    if not np.all(np.isfinite(linear_srgb)):
        raise ValueError("isp_linear_srgb contains NaN or Inf")

    expected_shape = metadata.get("isp_linear_srgb_shape")
    if expected_shape is not None and tuple(expected_shape) != linear_srgb.shape:
        raise ValueError(
            "Metadata shape does not match isp_linear_srgb.npy: "
            f"metadata={tuple(expected_shape)}, array={linear_srgb.shape}"
        )
    return linear_path, linear_srgb.astype(np.float32, copy=False)


def save_srgb_png(srgb, output_path, bit_depth):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if bit_depth == 8:
        image = np.rint(np.clip(srgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    elif bit_depth == 16:
        image = np.rint(np.clip(srgb, 0.0, 1.0) * 65535.0).astype(np.uint16)
    else:
        raise ValueError("bit_depth must be 8 or 16")

    if not cv2.imwrite(str(output_path), image[..., ::-1]):
        raise OSError(f"Failed to write image: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Render isp_linear_srgb.npy as a gamma-encoded sRGB PNG"
    )
    parser.add_argument(
        "--input-npy",
        required=True,
        help="Path to isp_linear_srgb.npy",
    )
    parser.add_argument(
        "--metadata-json",
        required=True,
        help="Path to the matching metadata.json",
    )
    parser.add_argument(
        "--output",
        help="Output PNG path; defaults to reconstructed_srgb.png beside metadata.json",
    )
    parser.add_argument("--bit-depth", type=int, choices=(8, 16), default=8)
    args = parser.parse_args()

    metadata_path, metadata = load_metadata(args.metadata_json)
    linear_path, linear_srgb = load_linear_srgb(args.input_npy, metadata)
    srgb = linear_to_srgb(linear_srgb)
    output_path = (
        Path(args.output).resolve()
        if args.output is not None
        else metadata_path.parent / "reconstructed_srgb.png"
    )
    save_srgb_png(srgb, output_path, args.bit_depth)

    clipped_fraction = float(np.mean((linear_srgb < 0.0) | (linear_srgb > 1.0)))
    print(f"metadata: {metadata_path}")
    print(f"linear sRGB: {linear_path}")
    print(f"shape: {linear_srgb.shape}")
    print(f"linear range: [{float(np.min(linear_srgb)):.6f}, {float(np.max(linear_srgb)):.6f}]")
    print(f"fraction clipped for display: {clipped_fraction:.8f}")
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
