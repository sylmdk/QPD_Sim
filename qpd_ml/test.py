from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lightning.pytorch.loggers import TensorBoardLogger
from PIL import Image
from torch.nn import functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure

from qpd_ml import QPDLightningModule
from qpd_ml.color import camera_rgb_to_srgb
from qpd_ml.config import copy_section, load_config
from qpd_ml.data import normalize_qpd_raw, pack_quad_bayer


@torch.inference_mode()
def reconstruct_full_image(
    model: torch.nn.Module,
    raw: np.ndarray,
    metadata: dict[str, Any],
) -> np.ndarray:
    """Run the entire QPD image through the model and return HWC camera RGB."""
    if raw.ndim != 2 or raw.shape[0] % 2 or raw.shape[1] % 2:
        raise ValueError(f"QPD RAW must be an even-sized 2D array, got {raw.shape}")
    normalized = normalize_qpd_raw(raw, metadata)
    packed = pack_quad_bayer(normalized).copy()
    device = next(model.parameters()).device
    inputs = torch.from_numpy(packed).unsqueeze(0).to(device=device, dtype=torch.float32)
    model.eval()
    prediction = model(inputs)[0].detach().float().cpu().numpy()
    return np.moveaxis(prediction, 0, -1)


def _camera_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    writable = np.array(image, dtype=np.float32, copy=True)
    return torch.from_numpy(writable).permute(2, 0, 1).unsqueeze(0).to(device)


def _color_metadata(metadata: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    wb = torch.tensor(metadata["isp_params"]["wb_gains"], dtype=torch.float32, device=device).unsqueeze(0)
    ccm = torch.tensor(
        metadata["isp_params"]["ccm_srgb_from_cam"], dtype=torch.float32, device=device
    ).unsqueeze(0)
    return wb, ccm


@torch.inference_mode()
def calculate_full_image_metrics(
    model: QPDLightningModule,
    prediction: np.ndarray,
    target: np.ndarray,
    metadata: dict[str, Any],
) -> dict[str, float]:
    """Calculate every metric from the same full-resolution prediction and target."""
    if prediction.shape != target.shape:
        raise ValueError(f"Prediction/target shape mismatch: {prediction.shape} vs {target.shape}")
    device = next(model.parameters()).device
    pred = _camera_tensor(prediction, device)
    truth = _camera_tensor(target, device)
    l1 = F.l1_loss(pred, truth)
    mse = F.mse_loss(pred, truth)
    loss = float(model.loss_config.get("l1_weight", 1.0)) * l1
    if model.loss_config.get("mse_weight", 0.0):
        loss = loss + float(model.loss_config["mse_weight"]) * mse

    domain = model.metrics_config.get("domain", "srgb")
    if domain == "srgb":
        wb, ccm = _color_metadata(metadata, device)
        metric_pred = camera_rgb_to_srgb(pred, wb, ccm)
        metric_target = camera_rgb_to_srgb(truth, wb, ccm)
    elif domain == "camera_linear":
        metric_pred = pred.clamp(0.0, 1.0)
        metric_target = truth.clamp(0.0, 1.0)
    else:
        raise ValueError("metrics.domain must be 'srgb' or 'camera_linear'")

    metric_mse = F.mse_loss(metric_pred, metric_target)
    psnr = -10.0 * torch.log10(metric_mse.clamp_min(1e-10))
    ssim = StructuralSimilarityIndexMeasure(
        data_range=float(model.metrics_config.get("data_range", 1.0))
    ).to(device)(metric_pred, metric_target)
    result = {
        "test_loss": float(loss),
        "test_l1": float(l1),
        "test_psnr": float(psnr),
        "test_ssim": float(ssim),
    }
    if model.lpips_enabled:
        model.lpips_metric.reset()
        result["test_lpips"] = float(model.lpips_metric(metric_pred, metric_target))
    return result


def save_srgb_png(camera_rgb: np.ndarray, metadata: dict[str, Any], path: Path) -> None:
    image = _camera_tensor(camera_rgb, torch.device("cpu"))
    wb, ccm = _color_metadata(metadata, torch.device("cpu"))
    srgb = camera_rgb_to_srgb(image, wb, ccm)[0].permute(1, 2, 0).numpy()
    png = np.rint(np.clip(srgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(png, mode="RGB").save(path)


def run_test_samples(
    model: QPDLightningModule,
    data_dir: str | Path,
    data_split: str,
    test_config: dict[str, Any],
    inference_config: dict[str, Any],
    shard_rank: int = 0,
    shard_count: int = 1,
    write_root_summary: bool = True,
) -> list[dict[str, Any]]:
    run_metrics = bool(test_config.get("run_metrics", True))
    save_outputs = bool(inference_config.get("enabled", True))
    if not run_metrics and not save_outputs:
        raise ValueError("Enable test.run_metrics or inference.enabled")
    save_clean = save_outputs and bool(inference_config.get("save_clean_energy_field", True))
    save_png = save_outputs and bool(inference_config.get("save_srgb_png", True))
    if save_outputs and not save_clean and not save_png:
        raise ValueError("Enable at least one inference output format")

    root = Path(data_dir).resolve()
    with (root / f"{data_split}.csv").open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    selected = set(inference_config.get("sample_ids") or [])
    if selected:
        rows = [row for row in rows if row["sample_id"] in selected]
        missing = selected.difference(row["sample_id"] for row in rows)
        if missing:
            raise ValueError(f"sample_ids not found in {data_split}.csv: {sorted(missing)}")
    rows = rows[shard_rank::shard_count]

    output_root = Path(inference_config.get("output_dir", "outputs/qpd_reconstruction"))
    if save_outputs:
        output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for row in rows:
        started = time.perf_counter()
        raw_path = root / Path(row["input_qpd_raw"])
        target_path = root / Path(row["target_clean_energy"])
        metadata_path = root / Path(row["metadata"])
        raw = np.load(raw_path, mmap_mode="r", allow_pickle=False)
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        prediction = reconstruct_full_image(model, raw, metadata)
        metrics = None
        if run_metrics:
            target = np.load(target_path, mmap_mode="r", allow_pickle=False)
            metrics = calculate_full_image_metrics(model, prediction, target, metadata)

        sample_dir = output_root / row["sample_id"]
        clean_path = sample_dir / "clean_energy_field_pred.npy"
        png_path = sample_dir / "reconstructed_srgb.png"
        if save_outputs:
            sample_dir.mkdir(parents=True, exist_ok=True)
            if save_clean:
                np.save(clean_path, prediction.astype(np.float32, copy=False), allow_pickle=False)
            if save_png:
                save_srgb_png(prediction, metadata, png_path)
        summary = {
            "sample_id": row["sample_id"],
            "input_qpd_raw": str(raw_path),
            "target_clean_energy": str(target_path) if run_metrics else None,
            "shape": list(prediction.shape),
            "range": [float(prediction.min()), float(prediction.max())],
            "metrics": metrics,
            "clean_energy_field": str(clean_path) if save_clean else None,
            "srgb_png": str(png_path) if save_png else None,
            "elapsed_seconds": time.perf_counter() - started,
        }
        if save_outputs:
            with (sample_dir / "reconstruction.json").open("w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    if save_outputs and write_root_summary:
        with (output_root / "reconstruction_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=2, ensure_ascii=False)
    return summaries


def _resolve_gpu_ids(trainer_config: dict[str, Any]) -> list[int]:
    """Resolve Lightning-style accelerator/devices into exact CUDA indices."""
    accelerator = str(trainer_config.get("accelerator", "auto")).lower()
    if accelerator == "cpu":
        return []
    if not torch.cuda.is_available():
        if accelerator in ("gpu", "cuda"):
            raise RuntimeError("GPU acceleration was requested, but CUDA is unavailable")
        return []
    available = torch.cuda.device_count()
    requested = trainer_config.get("devices", 1)
    if requested in ("auto", -1):
        gpu_ids = list(range(available))
    elif isinstance(requested, int):
        if requested < 1:
            raise ValueError("trainer.devices must be positive")
        gpu_ids = list(range(requested))
    elif isinstance(requested, (list, tuple)):
        gpu_ids = [int(device) for device in requested]
    elif isinstance(requested, str):
        gpu_ids = [int(device.strip()) for device in requested.split(",") if device.strip()]
    else:
        raise TypeError("trainer.devices must be an integer, list, comma string, or 'auto'")
    if not gpu_ids or len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("trainer.devices must contain unique GPU indices")
    invalid = [device for device in gpu_ids if device < 0 or device >= available]
    if invalid:
        raise ValueError(f"Requested unavailable GPU indices {invalid}; available range is 0..{available - 1}")
    return gpu_ids


def _load_model(checkpoint: str, metrics: dict[str, Any], device: torch.device) -> QPDLightningModule:
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model = QPDLightningModule.load_from_checkpoint(checkpoint, metrics=metrics)
    return model.to(device).eval()


def _distributed_test_worker(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """One spawned process owns one GPU and a disjoint sample shard."""
    device = torch.device(f"cuda:{payload['gpu_id']}")
    model = _load_model(payload["checkpoint"], payload["metrics"], device)
    return run_test_samples(
        model=model,
        data_dir=payload["data_dir"],
        data_split=payload["data_split"],
        test_config=payload["test_config"],
        inference_config=payload["inference_config"],
        shard_rank=payload["shard_rank"],
        shard_count=payload["shard_count"],
        write_root_summary=False,
    )


def _write_root_summary(summaries: list[dict[str, Any]], inference_config: dict[str, Any]) -> None:
    if not inference_config.get("enabled", True):
        return
    output_root = Path(inference_config.get("output_dir", "outputs/qpd_reconstruction"))
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "reconstruction_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)


def _mean_metrics(summaries: list[dict[str, Any]]) -> dict[str, float]:
    metrics = [item["metrics"] for item in summaries if item["metrics"] is not None]
    if not metrics:
        return {}
    return {key: float(np.mean([item[key] for item in metrics])) for key in metrics[0]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate and reconstruct full-size QPD data.")
    parser.add_argument("--config", type=Path, default=Path("qpd_ml/configs/test.yaml"))
    parser.add_argument("--checkpoint", type=Path, help="Override test.checkpoint in YAML")
    args = parser.parse_args()
    config = load_config(
        args.config,
        required_sections=("experiment", "data", "metrics", "trainer", "test", "inference"),
    )
    test_config = copy_section(config, "test")
    checkpoint = args.checkpoint or test_config.get("checkpoint")
    if checkpoint is None:
        raise ValueError("Set test.checkpoint in test.yaml or pass --checkpoint PATH")

    trainer_config = copy_section(config, "trainer")
    data_config = copy_section(config, "data")
    metrics_config = copy_section(config, "metrics")
    inference_config = copy_section(config, "inference")
    gpu_ids = _resolve_gpu_ids(trainer_config)
    if len(gpu_ids) > 1:
        payloads = [
            {
                "gpu_id": gpu_id,
                "shard_rank": rank,
                "shard_count": len(gpu_ids),
                "checkpoint": str(checkpoint),
                "metrics": metrics_config,
                "data_dir": data_config["data_dir"],
                "data_split": data_config["data_split"],
                "test_config": test_config,
                "inference_config": inference_config,
            }
            for rank, gpu_id in enumerate(gpu_ids)
        ]
        context = mp.get_context("spawn")
        with context.Pool(processes=len(gpu_ids)) as pool:
            worker_results = pool.map(_distributed_test_worker, payloads)
        summaries = sorted(
            [summary for worker in worker_results for summary in worker],
            key=lambda item: item["sample_id"],
        )
        _write_root_summary(summaries, inference_config)
    else:
        device = torch.device(f"cuda:{gpu_ids[0]}" if gpu_ids else "cpu")
        model = _load_model(str(checkpoint), metrics_config, device)
        summaries = run_test_samples(
            model, data_config["data_dir"], data_config["data_split"], test_config,
            inference_config,
        )
    mean_metrics = _mean_metrics(summaries)
    if mean_metrics:
        experiment = copy_section(config, "experiment")
        logger = TensorBoardLogger(
            save_dir=str(experiment.get("output_dir", "outputs/qpd_training")),
            name=str(experiment.get("name", "qpd_test")), version=experiment.get("version"),
            default_hp_metric=False,
        )
        logger.log_metrics(mean_metrics, step=0)
        logger.finalize("success")
        print("Full-resolution mean metrics:")
        print(json.dumps(mean_metrics, indent=2))


if __name__ == "__main__":
    main()
