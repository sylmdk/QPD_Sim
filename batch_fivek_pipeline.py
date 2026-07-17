import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
import urllib.request
import urllib.parse
from pathlib import Path


FIVEK_BASE_URL = "https://groups.csail.mit.edu/graphics/fivek_dataset/img/dng"
FIVEK_DATASET_URL = "https://groups.csail.mit.edu/graphics/fivek_dataset/"
DEFAULT_FIVEK_SAMPLES = [
    "a4207-kme_1045.dng",
    "a4210-kme_0540.dng",
]
QPD_CFA_PATTERN = "RGGB"
QPD_CFA_LAYOUT = "quad_bayer_2x2_blocks"
QPD_SIMULATOR_TYPE = "hwk_full_field"
QPD_SIMULATOR_VERSION = 1


def sha256_file(path):
    if path is None:
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def list_fivek_dng_names():
    html = urllib.request.urlopen(FIVEK_DATASET_URL, timeout=120).read().decode("utf-8", "ignore")
    names = re.findall(r'href="img/dng/([^"]+\.dng)"', html)
    seen = set()
    unique_names = []
    for name in names:
        if name not in seen:
            seen.add(name)
            unique_names.append(name)
    if not unique_names:
        raise RuntimeError("Could not find FiveK DNG links on the official dataset page.")
    return unique_names


def download_file(url, dst_path, retries=3, sleep_seconds=3.0):
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.exists() and dst_path.stat().st_size > 0:
        return "exists"

    tmp_path = dst_path.with_suffix(dst_path.suffix + ".part")
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            safe_url = urllib.parse.quote(url, safe=':/')
            with urllib.request.urlopen(safe_url, timeout=120) as response:
                with open(tmp_path, "wb") as f:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
            tmp_path.replace(dst_path)
            return "downloaded"
        except Exception as exc:
            last_error = exc
            if tmp_path.exists():
                tmp_path.unlink()
            if attempt < retries:
                time.sleep(sleep_seconds)
    raise RuntimeError(f"Failed to download {url}: {last_error}") from last_error


def download_samples(raw_dir, sample_names):
    records = []
    for name in sample_names:
        url = f"{FIVEK_BASE_URL}/{name}"
        dst_path = raw_dir / name
        status = download_file(url, dst_path)
        records.append({"name": name, "url": url, "path": str(dst_path), "status": status})
        print(f"{status}: {dst_path}")
    return records


def output_is_complete(output_dir, args):
    required = ("metadata.json", "qpd_raw.npy", "clean_energy_field.npy")
    if not all((output_dir / name).exists() for name in required):
        return False
    try:
        with open(output_dir / "metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    expected = {
        "crop": args.crop,
        "qpd_readout_mode": "same",
        "qpd_cfa_pattern": QPD_CFA_PATTERN,
        "qpd_cfa_layout": QPD_CFA_LAYOUT,
        "qpd_simulator_type": "disabled" if args.skip_qpd_sim else QPD_SIMULATOR_TYPE,
        "qpd_simulator_version": QPD_SIMULATOR_VERSION,
    }
    if not all(metadata.get(key) == value for key, value in expected.items()):
        return False
    if metadata.get("isp_params", {}).get("ccm_source_requested") != args.ccm_source:
        return False
    if args.skip_qpd_sim:
        return True
    request = metadata.get("qpd_simulation_request", {})
    expected_request = {
        "hwk_dir": str(Path(args.hwk_dir).resolve()),
        "hwk_config": None if args.hwk_config is None else str(Path(args.hwk_config).resolve()),
        "hwk_config_sha256": sha256_file(args.hwk_config),
        "distance": args.hwk_distance,
        "aperture": args.hwk_aperture,
        "cache_enabled": not args.no_hwk_cache,
    }
    return request == expected_request


def run_single_image(pipeline_path, raw_path, output_root, args, index):
    stem = raw_path.stem
    output_dir = output_root / stem
    if args.skip_existing and output_is_complete(output_dir, args):
        print(f"skip existing output: {output_dir}")
        with open(output_dir / "metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return summarize_result(raw_path, output_dir, metadata, skipped=True)

    cmd = [
        sys.executable,
        str(pipeline_path),
        "--input",
        str(raw_path),
        "--input-kind",
        "raw",
        "--output-dir",
        str(output_dir),
        "--noise-table",
        str(args.noise_table),
        "--seed",
        str(args.seed + index),
    ]

    if args.crop:
        cmd.extend(["--crop", args.crop])
    if args.ccm_source:
        cmd.extend(["--ccm-source", args.ccm_source])
    if args.hwk_dir:
        cmd.extend(["--hwk-dir", str(args.hwk_dir)])
    if args.hwk_config:
        cmd.extend(["--hwk-config", str(args.hwk_config)])
    if args.hwk_distance:
        cmd.extend(["--hwk-distance", args.hwk_distance])
    if args.hwk_aperture:
        cmd.extend(["--hwk-aperture", args.hwk_aperture])
    if args.no_hwk_cache:
        cmd.append("--no-hwk-cache")
    if args.skip_qpd_sim:
        cmd.append("--skip-qpd-sim")
    if args.skip_noise:
        cmd.append("--skip-noise")

    print(f"processing: {raw_path}")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        message = (exc.stderr or exc.stdout or str(exc)).strip()
        print(f"failed: {raw_path}")
        print(message)
        if args.fail_fast:
            raise
        output_dir.mkdir(parents=True, exist_ok=True)
        failure = {
            "input": str(raw_path),
            "output_dir": str(output_dir),
            "failed": True,
            "returncode": exc.returncode,
            "error": message,
        }
        with open(output_dir / "failed.json", "w", encoding="utf-8") as f:
            json.dump(failure, f, indent=2)
        return failure

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return summarize_result(raw_path, output_dir, metadata, skipped=False)


def summarize_result(raw_path, output_dir, metadata, skipped=False):
    return {
        "input": str(raw_path),
        "output_dir": str(output_dir),
        "skipped_existing": skipped,
        "shape": metadata.get("shape"),
        "qpd_output_levels": metadata.get("qpd_output_levels"),
        "iso": metadata.get("isp_params", {}).get("iso"),
        "iso_source": metadata.get("isp_params", {}).get("iso_source"),
        "noise_iso": None if metadata.get("noise_row") is None else metadata["noise_row"].get("ISO"),
        "noise_iso_selection": None
        if metadata.get("noise_row") is None
        else metadata["noise_row"].get("noise_iso_selection"),
        "ccm_source": metadata.get("isp_params", {}).get("ccm_source"),
        "qpd_readout_mode": metadata.get("qpd_readout_mode"),
        "qpd_cfa_pattern": metadata.get("qpd_cfa_pattern"),
        "qpd_cfa_layout": metadata.get("qpd_cfa_layout"),
        "qpd_simulator_type": metadata.get("qpd_simulator_type"),
        "hwk_condition": None
        if metadata.get("qpd_simulation") is None
        else {
            "distance": metadata["qpd_simulation"].get("distance"),
            "aperture": metadata["qpd_simulation"].get("aperture"),
        },
        "rdm_mix": None
        if metadata.get("qpd_simulation") is None
        else metadata["qpd_simulation"].get("rdm_mix"),
        "roundtrip_error": metadata.get("reversible_isp_roundtrip_error"),
    }


def main():
    parser = argparse.ArgumentParser(description="Batch process MIT-Adobe FiveK DNG files through the HWK QPD pipeline")
    parser.add_argument("--raw-dir", default="data/raw_samples/fivek_batch", help="Directory containing FiveK DNG files")
    parser.add_argument("--output-root", default="outputs/fivek_batch", help="Root directory for per-image outputs")
    parser.add_argument("--pipeline", default="qpd_qsc_pipeline.py", help="Single-image pipeline script")
    parser.add_argument("--noise-table", default="noise_table.csv")
    parser.add_argument("--crop", default="3000x2000")
    parser.add_argument("--hwk-dir", help="HWK field_data directory or statistics root")
    parser.add_argument("--hwk-config", help="Optional HWK/RDM simulator config JSON")
    parser.add_argument("--hwk-distance", help="Select one calibrated object distance")
    parser.add_argument("--hwk-aperture", help="Select one calibrated aperture")
    parser.add_argument("--no-hwk-cache", action="store_true", help="Disable adjacent .csv.npz HWK caches")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--limit", type=int, help="Process at most N DNG files")
    parser.add_argument("--download-samples", action="store_true", help="Download two bundled FiveK test DNGs first")
    parser.add_argument("--download-all", action="store_true", help="Download all FiveK DNG files listed on the official page")
    parser.add_argument("--download-only", action="store_true", help="Only download DNG files; do not run the processing pipeline")
    parser.add_argument("--skip-existing", action="store_true", default=True, help="Skip images whose output metadata and qpd_raw already exist")
    parser.add_argument("--no-skip-existing", action="store_false", dest="skip_existing", help="Reprocess images even if outputs already exist")
    parser.add_argument("--fail-fast", action="store_true", help="Stop the batch on the first failed image")
    parser.add_argument(
        "--ccm-source",
        choices=("auto", "rawpy-fit", "metadata", "identity"),
        default="metadata",
        help="CCM source passed to qpd_qsc_pipeline.py; default matches the single-image pipeline.",
    )
    parser.add_argument("--skip-qpd-sim", action="store_true", help="Skip HWK/RDM while retaining Quad RGGB sampling")
    parser.add_argument("--skip-noise", action="store_true")
    args = parser.parse_args()

    if not args.download_only and not args.skip_qpd_sim and args.hwk_dir is None:
        parser.error("--hwk-dir is required for processing unless --skip-qpd-sim is used")

    raw_dir = Path(args.raw_dir)
    output_root = Path(args.output_root)
    pipeline_path = Path(args.pipeline)
    output_root.mkdir(parents=True, exist_ok=True)

    download_records = []
    if args.download_all:
        names = list_fivek_dng_names()
        if args.limit is not None:
            names = names[: args.limit]
        manifest_path = raw_dir / "fivek_download_manifest.json"
        raw_dir.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "source": FIVEK_DATASET_URL,
                    "count": len(names),
                    "names": names,
                },
                f,
                indent=2,
            )
        print(f"found {len(names)} FiveK DNG files")
        download_records.extend(download_samples(raw_dir, names))

    if args.download_samples:
        download_records.extend(download_samples(raw_dir, DEFAULT_FIVEK_SAMPLES))

    if args.download_only:
        summary = {
            "raw_dir": str(raw_dir),
            "output_root": str(output_root),
            "crop": args.crop,
            "count": 0,
            "downloads": download_records,
            "results": [],
        }
        summary_path = output_root / "batch_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"download only; saved summary: {summary_path}")
        return

    raw_paths = sorted(raw_dir.glob("*.dng"))
    if args.limit is not None:
        raw_paths = raw_paths[: args.limit]
    if not raw_paths:
        raise FileNotFoundError(f"No .dng files found in {raw_dir}")

    results = []
    for index, raw_path in enumerate(raw_paths):
        results.append(run_single_image(pipeline_path, raw_path, output_root, args, index))

    summary = {
        "raw_dir": str(raw_dir),
        "output_root": str(output_root),
        "crop": args.crop,
        "hwk_dir": None if args.hwk_dir is None else str(Path(args.hwk_dir).resolve()),
        "hwk_config": None if args.hwk_config is None else str(Path(args.hwk_config).resolve()),
        "hwk_distance": args.hwk_distance,
        "hwk_aperture": args.hwk_aperture,
        "hwk_cache_enabled": not args.no_hwk_cache,
        "count": len(results),
        "success_count": sum(1 for result in results if not result.get("failed")),
        "failed_count": sum(1 for result in results if result.get("failed")),
        "skipped_existing_count": sum(1 for result in results if result.get("skipped_existing")),
        "downloads": download_records,
        "results": results,
    }
    summary_path = output_root / "batch_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"saved summary: {summary_path}")


if __name__ == "__main__":
    main()
