from __future__ import annotations

import argparse
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from qpd_ml import QPDDataModule, QPDLightningModule
from qpd_ml.config import copy_section, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train QPD reconstruction from a YAML config.")
    parser.add_argument(
        "--config", type=Path, default=Path("qpd_ml/configs/train.yaml"),
        help="Experiment YAML. All training/model/data options are read from this file.",
    )
    return parser.parse_args()


def resolve_resume_checkpoint(
    resume_config: dict,
    experiment_config: dict,
) -> Path | None:
    """Resolve an explicit checkpoint or the newest last.ckpt for this experiment."""
    if not bool(resume_config.get("enabled", False)):
        return None
    requested = resume_config.get("checkpoint", "auto")
    if requested not in (None, "", "auto"):
        checkpoint = Path(requested).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {checkpoint}")
        return checkpoint

    output_dir = Path(experiment_config.get("output_dir", "outputs/qpd_training"))
    experiment_name = str(experiment_config.get("name", "qpd_unet"))
    experiment_root = (output_dir / experiment_name).resolve()
    candidates = list(experiment_root.glob("*/checkpoints/last.ckpt"))
    candidates.extend(experiment_root.glob("checkpoints/last.ckpt"))
    if not candidates:
        raise FileNotFoundError(
            f"resume.enabled is true, but no last.ckpt was found under {experiment_root}"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _resume_logger_version(
    checkpoint: Path | None,
    output_dir: Path,
    experiment_name: str,
    configured_version,
):
    """Reuse the original TensorBoard version when resuming an auto-versioned run."""
    if checkpoint is None or configured_version is not None:
        return configured_version
    run_dir = checkpoint.parent.parent
    expected_parent = (output_dir / experiment_name).resolve()
    if run_dir.parent.resolve() == expected_parent:
        return run_dir.name
    return configured_version


def main() -> None:
    config = load_config(
        parse_args().config,
        required_sections=(
            "experiment", "data", "model", "optimizer", "loss", "metrics",
            "visualization", "checkpoint", "trainer", "resume",
        ),
    )
    experiment = copy_section(config, "experiment")
    seed = int(experiment.get("seed", 2026))
    L.seed_everything(seed, workers=True)

    data_config = copy_section(config, "data")
    data_config.setdefault("seed", seed)
    datamodule = QPDDataModule(**data_config)
    model = QPDLightningModule(
        model=copy_section(config, "model"),
        optimizer=copy_section(config, "optimizer"),
        loss=copy_section(config, "loss"),
        metrics=copy_section(config, "metrics"),
        visualization=copy_section(config, "visualization"),
    )

    output_dir = Path(experiment.get("output_dir", "outputs/qpd_training"))
    experiment_name = str(experiment.get("name", "unet"))
    resume_checkpoint = resolve_resume_checkpoint(copy_section(config, "resume"), experiment)
    logger_version = _resume_logger_version(
        resume_checkpoint, output_dir, experiment_name, experiment.get("version")
    )
    logger = TensorBoardLogger(
        save_dir=str(output_dir), name=experiment_name,
        version=logger_version, default_hp_metric=False,
    )
    checkpoint_config = copy_section(config, "checkpoint")
    monitor = str(checkpoint_config.get("monitor", "val_psnr"))
    checkpoint = ModelCheckpoint(
        dirpath=str(Path(logger.log_dir) / "checkpoints"),
        filename=str(checkpoint_config.get("filename", "qpd-{epoch:03d}-{val_psnr:.2f}")),
        monitor=monitor, mode=str(checkpoint_config.get("mode", "max")),
        save_last=bool(checkpoint_config.get("save_last", True)),
        save_top_k=int(checkpoint_config.get("save_top_k", 1)),
    )
    trainer_config = copy_section(config, "trainer")
    trainer = L.Trainer(
        logger=logger, callbacks=[checkpoint, LearningRateMonitor(logging_interval="epoch")],
        default_root_dir=str(output_dir), **trainer_config,
    )
    if resume_checkpoint is not None:
        print(f"Resuming training from: {resume_checkpoint}")
    trainer.fit(
        model, datamodule=datamodule,
        ckpt_path=str(resume_checkpoint) if resume_checkpoint is not None else None,
    )


if __name__ == "__main__":
    main()
