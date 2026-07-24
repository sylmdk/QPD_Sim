import argparse
import csv
import json
import random
import shutil
from pathlib import Path


REQUIRED_FILES = {
    "input": "qpd_raw.npy",
    "target": "clean_energy_field.npy",
    "metadata": "metadata.json",
}
QPD_CFA_PATTERN = "RGGB"
QPD_CFA_LAYOUT = "quad_bayer_2x2_blocks"
QPD_SIMULATOR_TYPE = "hwk_full_field"
QPD_SIMULATOR_VERSION = 1


def is_valid_sample(sample_dir):
    if (sample_dir / "failed.json").exists():
        return False
    return all((sample_dir / filename).exists() for filename in REQUIRED_FILES.values())


def collect_samples(source_root):
    source_root = Path(source_root)
    samples = []
    for sample_dir in sorted(path for path in source_root.iterdir() if path.is_dir()):
        if not is_valid_sample(sample_dir):
            continue
        metadata_path = sample_dir / REQUIRED_FILES["metadata"]
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        if metadata.get("qpd_cfa_pattern") != QPD_CFA_PATTERN:
            continue
        if metadata.get("qpd_cfa_layout") != QPD_CFA_LAYOUT:
            continue
        if metadata.get("qpd_simulator_type") != QPD_SIMULATOR_TYPE:
            continue
        if metadata.get("qpd_simulator_version") != QPD_SIMULATOR_VERSION:
            continue
        samples.append(
            {
                "sample_id": sample_dir.name,
                "sample_dir": sample_dir,
                "input": sample_dir / REQUIRED_FILES["input"],
                "target": sample_dir / REQUIRED_FILES["target"],
                "metadata": metadata_path,
                "shape": metadata.get("shape"),
                "qpd_shape": metadata.get("qpd_raw_shape", metadata.get("shape")),
                "target_shape": metadata.get("clean_energy_shape"),
                "target_dtype": metadata.get("clean_energy_dtype"),
                "iso": metadata.get("isp_params", {}).get("iso"),
                "noise_iso": None if metadata.get("noise_row") is None else metadata["noise_row"].get("ISO"),
                "ccm_source": metadata.get("isp_params", {}).get("ccm_source"),
                "qpd_readout_mode": metadata.get("qpd_readout_mode"),
                "qpd_cfa_pattern": metadata.get("qpd_cfa_pattern"),
                "qpd_cfa_layout": metadata.get("qpd_cfa_layout"),
                "qpd_simulator_type": metadata.get("qpd_simulator_type"),
                "hwk_distance": metadata.get("qpd_simulation", {}).get("distance"),
                "hwk_aperture": metadata.get("qpd_simulation", {}).get("aperture"),
            }
        )
    return samples


def split_counts(total, train_ratio, val_ratio):
    ratios = [train_ratio, val_ratio, max(0.0, 1.0 - train_ratio - val_ratio)]
    raw_counts = [total * ratio for ratio in ratios]
    counts = [int(count) for count in raw_counts]
    remainder = total - sum(counts)
    order = sorted(range(3), key=lambda idx: raw_counts[idx] - counts[idx], reverse=True)
    for idx in order[:remainder]:
        counts[idx] += 1

    nonzero = [idx for idx, ratio in enumerate(ratios) if ratio > 0]
    if total >= len(nonzero):
        for idx in nonzero:
            if counts[idx] == 0:
                donor = max((i for i in range(3) if counts[i] > 1), key=lambda i: counts[i], default=None)
                if donor is not None:
                    counts[donor] -= 1
                    counts[idx] += 1
    return counts[0], counts[1], counts[2]


def split_samples(samples, train_ratio, val_ratio, seed, num_samples=None, split_order="random"):
    selected = list(samples)
    if split_order == "random":
        rng = random.Random(seed)
        rng.shuffle(selected)
    elif split_order != "sequential":
        raise ValueError(f"Unsupported split_order: {split_order}")

    if num_samples is not None:
        if num_samples <= 0:
            raise ValueError("num_samples must be a positive integer")
        if num_samples > len(selected):
            raise ValueError(
                f"num_samples ({num_samples}) exceeds the number of valid samples ({len(selected)})"
            )
        selected = selected[:num_samples]

    train_count, val_count, _ = split_counts(len(selected), train_ratio, val_ratio)
    return {
        "train": selected[:train_count],
        "val": selected[train_count:train_count + val_count],
        "test": selected[train_count + val_count:],
    }


def make_relative(path, root):
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def materialize_file(src, dst, mode):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        try:
            dst.hardlink_to(src)
        except OSError:
            shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unsupported materialize mode: {mode}")


def materialize_split(split_name, samples, output_root, mode):
    for sample in samples:
        sample_out = output_root / split_name / sample["sample_id"]
        materialize_file(sample["input"], sample_out / REQUIRED_FILES["input"], mode)
        materialize_file(sample["target"], sample_out / REQUIRED_FILES["target"], mode)
        materialize_file(sample["metadata"], sample_out / REQUIRED_FILES["metadata"], mode)


def manifest_rows(split_name, samples, manifest_root, materialized):
    rows = []
    for sample in samples:
        if materialized:
            base = manifest_root / split_name / sample["sample_id"]
            input_path = base / REQUIRED_FILES["input"]
            target_path = base / REQUIRED_FILES["target"]
            metadata_path = base / REQUIRED_FILES["metadata"]
        else:
            input_path = sample["input"]
            target_path = sample["target"]
            metadata_path = sample["metadata"]
        rows.append(
            {
                "split": split_name,
                "sample_id": sample["sample_id"],
                "input_qpd_raw": make_relative(input_path, manifest_root),
                "target_clean_energy": make_relative(target_path, manifest_root),
                "metadata": make_relative(metadata_path, manifest_root),
                "shape": json.dumps(sample["shape"]),
                "qpd_shape": json.dumps(sample["qpd_shape"]),
                "target_shape": json.dumps(sample["target_shape"]),
                "target_dtype": sample["target_dtype"],
                "iso": sample["iso"],
                "noise_iso": sample["noise_iso"],
                "ccm_source": sample["ccm_source"],
                "qpd_readout_mode": sample["qpd_readout_mode"],
                "qpd_cfa_pattern": sample["qpd_cfa_pattern"],
                "qpd_cfa_layout": sample["qpd_cfa_layout"],
                "qpd_simulator_type": sample["qpd_simulator_type"],
                "hwk_distance": sample["hwk_distance"],
                "hwk_aperture": sample["hwk_aperture"],
            }
        )
    return rows


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "sample_id",
        "input_qpd_raw",
        "target_clean_energy",
        "metadata",
        "shape",
        "qpd_shape",
        "target_shape",
        "target_dtype",
        "iso",
        "noise_iso",
        "ccm_source",
        "qpd_readout_mode",
        "qpd_cfa_pattern",
        "qpd_cfa_layout",
        "qpd_simulator_type",
        "hwk_distance",
        "hwk_aperture",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Create train/val/test splits for QPD raw -> clean energy field dataset")
    parser.add_argument("--source-root", required=True, help="Batch output root containing one directory per processed sample")
    parser.add_argument("--output-root", required=True, help="Directory for split manifests and optional materialized dataset")
    parser.add_argument("--train", type=float, default=0.8, help="Train ratio")
    parser.add_argument("--val", type=float, default=0.1, help="Validation ratio")
    parser.add_argument("--test", type=float, default=0.1, help="Test ratio; checked against train/val")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of valid samples to include; default uses all valid samples",
    )
    parser.add_argument(
        "--split-order",
        choices=("random", "sequential"),
        default="random",
        help="random shuffles with --seed; sequential preserves sorted sample-directory order",
    )
    parser.add_argument(
        "--materialize",
        choices=("none", "copy", "hardlink"),
        default="none",
        help="none writes manifests only; copy/hardlink creates split folders with qpd_raw, clean_energy_field, metadata",
    )
    args = parser.parse_args()

    ratio_sum = args.train + args.val + args.test
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum}")

    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    samples = collect_samples(source_root)
    if not samples:
        raise FileNotFoundError(f"No valid samples found in {source_root}")

    splits = split_samples(
        samples,
        args.train,
        args.val,
        args.seed,
        args.num_samples,
        args.split_order,
    )
    participating_count = sum(len(split_samples_list) for split_samples_list in splits.values())
    output_root.mkdir(parents=True, exist_ok=True)

    if args.materialize != "none":
        for split_name, split_samples_list in splits.items():
            materialize_split(split_name, split_samples_list, output_root, args.materialize)

    all_rows = []
    split_summary = {}
    materialized = args.materialize != "none"
    for split_name, split_samples_list in splits.items():
        rows = manifest_rows(split_name, split_samples_list, output_root, materialized)
        all_rows.extend(rows)
        split_summary[split_name] = {
            "count": len(split_samples_list),
            "samples": [sample["sample_id"] for sample in split_samples_list],
        }
        write_json(output_root / f"{split_name}.json", rows)
        write_csv(output_root / f"{split_name}.csv", rows)

    write_json(output_root / "all.json", all_rows)
    write_csv(output_root / "all.csv", all_rows)
    write_json(
        output_root / "split_summary.json",
        {
            "source_root": str(source_root),
            "output_root": str(output_root),
            "total_valid_samples": len(samples),
            "requested_num_samples": args.num_samples,
            "participating_samples": participating_count,
            "split_order": args.split_order,
            "ratios": {"train": args.train, "val": args.val, "test": args.test},
            "seed": args.seed,
            "materialize": args.materialize,
            "splits": split_summary,
        },
    )

    print(f"valid samples: {len(samples)}")
    print(f"participating samples: {participating_count}")
    print(f"split order: {args.split_order}")
    for split_name in ("train", "val", "test"):
        print(f"{split_name}: {len(splits[split_name])}")
    print(f"saved manifests to: {output_root}")


if __name__ == "__main__":
    main()
