import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from crosstalk_sim import QpdSimulate


DEFAULT_QSC_CONFIGS = {
    "prob_shading": 0.0,
    "prob_blur": 0.6,
    "pad": 1,
    "ksize": 21,
    "sigma": (0.2, 1.8),
    "prob_reorder": 0.9,
    "reorder_range": 0.35,
    "reorder_color": 0.10,
    "prob_rdm_mix": 0.5,
    "mix_energey": 0.25,
    "prob_flatten": 0.8,
}

DEFAULT_ISP_PARAMS = {
    "cfa_pattern": "RGGB",
    "black_level": 64,
    "white_level": 1023,
    "bit_depth": 10,
    "iso": 100,
    "wb_gains": [2.0, 1.0, 1.6],
    "ccm_srgb_from_cam": [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
}

QPD_OUTPUT_LEVELS = {
    "bit_depth": 10,
    "black_level": 64,
    "white_level": 1023,
}

STANDARD_BAYER_PATTERNS = {"RGGB", "BGGR", "GRBG", "GBRG"}
QPD_CFA_PATTERN = "RGGB"
QPD_CFA_LAYOUT = "quad_bayer_2x2_blocks"

def srgb_to_linear(srgb):
    srgb = np.clip(srgb, 0.0, 1.0)
    return np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(linear_rgb):
    linear_rgb = np.clip(linear_rgb, 0.0, 1.0)
    return np.where(linear_rgb <= 0.0031308, linear_rgb * 12.92, 1.055 * (linear_rgb ** (1.0 / 2.4)) - 0.055)


def load_srgb_image(path):
    bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    if bgr.ndim == 2:
        bgr = np.repeat(bgr[..., None], 3, axis=2)
    if bgr.shape[2] == 4:
        bgr = bgr[:, :, :3]
    rgb = bgr[..., ::-1].astype(np.float32)
    max_value = 65535.0 if rgb.max() > 255.0 else 255.0
    return rgb / max_value


def cfa_pattern_from_rawpy(raw):
    if raw.raw_pattern.shape != (2, 2):
        raise ValueError(f"Unsupported RAW pattern shape {raw.raw_pattern.shape}; expected a standard 2x2 Bayer CFA.")
    color_desc = raw.color_desc.decode("ascii")
    return "".join(color_desc[int(raw.raw_pattern[y, x])] for y in range(2) for x in range(2)).upper()


def wb_gains_from_rawpy(raw):
    wb = np.asarray(raw.camera_whitebalance, dtype=np.float32)
    if wb.size < 4 or not np.all(np.isfinite(wb[:4])) or np.max(wb[:4]) <= 0:
        wb = np.asarray(raw.daylight_whitebalance, dtype=np.float32)

    g = np.mean([wb[1], wb[3]]) if wb.size >= 4 and wb[3] > 0 else wb[1]
    if g <= 0:
        return list(DEFAULT_ISP_PARAMS["wb_gains"])
    return [float(wb[0] / g), 1.0, float(wb[2] / g)]


def validate_ccm_matrix(ccm, name="ccm_srgb_from_cam"):
    ccm = np.asarray(ccm, dtype=np.float32)
    if ccm.shape != (3, 3):
        raise ValueError(f"{name} must be a 3x3 matrix, got shape {ccm.shape}")
    if not np.all(np.isfinite(ccm)):
        raise ValueError(f"{name} contains NaN or Inf values")
    det = float(np.linalg.det(ccm))
    if abs(det) < 1e-8:
        raise ValueError(f"{name} is singular or nearly singular; determinant={det:.3e}")
    return ccm


def read_exif_iso(path):
    try:
        import exifread
    except ImportError:
        return None

    try:
        with open(path, "rb") as f:
            tags = exifread.process_file(f, details=False)
    except Exception:
        return None

    for key in ("EXIF ISOSpeedRatings", "EXIF PhotographicSensitivity", "Image ISOSpeedRatings"):
        value = tags.get(key)
        if value is None:
            continue
        try:
            raw_value = value.values[0] if hasattr(value, "values") else value
            return float(raw_value.num) / float(raw_value.den) if hasattr(raw_value, "num") else float(raw_value)
        except Exception:
            try:
                return float(str(value).split(",")[0].strip())
            except Exception:
                continue
    return None


def load_raw_with_isp_params(path, override_params=None):
    try:
        import rawpy
    except ImportError as exc:
        raise ImportError(
            "RAW/DNG input requires rawpy. Install rawpy, or use --input-kind srgb "
            "with --isp-json for the debug path."
        ) from exc

    with rawpy.imread(str(path)) as raw:
        raw_data = raw.raw_image_visible.astype(np.float32).copy()
        cfa_pattern = cfa_pattern_from_rawpy(raw)
        black_levels = [float(v) for v in raw.black_level_per_channel[:4]]
        cfa_black_levels = [
            float(raw.black_level_per_channel[int(raw.raw_pattern[y, x])])
            for y in range(2)
            for x in range(2)
        ]
        black_level = float(np.mean(black_levels)) if black_levels else 0.0
        white_level = float(raw.white_level) if raw.white_level is not None else float(np.max(raw_data))
        bit_depth = int(np.ceil(np.log2(max(white_level + 1.0, 2.0))))
        rawpy_matrix = np.asarray(raw.rgb_xyz_matrix, dtype=np.float32)
        has_rawpy_color_matrix = bool(rawpy_matrix.size > 0)

        isp_params = dict(DEFAULT_ISP_PARAMS)
        isp_params.update(
            {
                "source_kind": "raw",
                "cfa_pattern": cfa_pattern,
                "black_level": black_level,
                "black_level_per_channel": black_levels,
                "cfa_black_level_2x2": cfa_black_levels,
                "white_level": white_level,
                "bit_depth": bit_depth,
                "iso": None,
                "iso_source": "unavailable",
                "wb_gains": wb_gains_from_rawpy(raw),
                "has_rawpy_color_matrix": has_rawpy_color_matrix,
                "ccm_srgb_from_cam": DEFAULT_ISP_PARAMS["ccm_srgb_from_cam"],
            }
        )

        iso = getattr(raw, "iso_speed", None)
        if iso:
            isp_params["iso"] = float(iso)
            isp_params["iso_source"] = "rawpy.iso_speed"

    if isp_params.get("iso") is None:
        exif_iso = read_exif_iso(path)
        if exif_iso is not None:
            isp_params["iso"] = float(exif_iso)
            isp_params["iso_source"] = "exifread"

    if override_params:
        isp_params.update(override_params)
    return raw_data, isp_params


def load_rawpy_linear_srgb(path, crop_width=None, crop_height=None):
    try:
        import rawpy
    except ImportError as exc:
        raise ImportError("RAW/DNG linear sRGB rendering requires rawpy.") from exc

    with rawpy.imread(str(path)) as raw:
        srgb_u16 = raw.postprocess(
            use_camera_wb=True,
            output_color=rawpy.ColorSpace.sRGB,
            gamma=(1, 1),
            no_auto_bright=True,
            output_bps=16,
            user_flip=0,
        )
    linear_srgb = srgb_u16.astype(np.float32) / 65535.0
    return center_crop_even(linear_srgb, crop_width, crop_height)


def normalize_raw_mosaic(raw_data, isp_params):
    raw_data = raw_data.astype(np.float32)
    pattern = validate_cfa_pattern(isp_params["cfa_pattern"])
    black_default = float(isp_params["black_level"])
    white_level = float(isp_params["white_level"])
    black_levels = isp_params.get("cfa_black_level_2x2") or isp_params.get("black_level_per_channel")

    if black_levels and len(black_levels) >= 4:
        black_map = np.empty(raw_data.shape, dtype=np.float32)
        for y in range(2):
            for x in range(2):
                black_map[y::2, x::2] = float(black_levels[y * 2 + x])
    else:
        black_map = black_default

    denom = np.maximum(white_level - black_map, 1.0)
    return np.clip((raw_data - black_map) / denom, 0.0, 1.0).astype(np.float32)


def demosaic_raw_to_camera_rgb(raw_linear, cfa_pattern):
    pattern = validate_cfa_pattern(cfa_pattern)
    codes = {
        "RGGB": cv2.COLOR_BayerBG2RGB,
        "BGGR": cv2.COLOR_BayerRG2RGB,
        "GRBG": cv2.COLOR_BayerGB2RGB,
        "GBRG": cv2.COLOR_BayerGR2RGB,
    }
    if pattern not in codes:
        raise ValueError(f"Unsupported CFA pattern {pattern}; expected one of {sorted(codes)}.")
    raw_u16 = np.rint(np.clip(raw_linear, 0.0, 1.0) * 65535.0).astype(np.uint16)
    return (cv2.cvtColor(raw_u16, codes[pattern]).astype(np.float32) / 65535.0).clip(0.0, 1.0)


def apply_forward_isp_linear(camera_rgb, isp_params):
    wb_gains = np.asarray(isp_params["wb_gains"], dtype=np.float32).reshape(1, 1, 3)
    balanced = camera_rgb * wb_gains
    ccm = validate_ccm_matrix(isp_params["ccm_srgb_from_cam"])
    return (balanced @ ccm.T).astype(np.float32)


def inverse_linear_isp_to_camera_rgb(linear_srgb, isp_params):
    ccm = validate_ccm_matrix(isp_params["ccm_srgb_from_cam"])
    cam_from_srgb = np.linalg.inv(ccm)
    balanced = linear_srgb @ cam_from_srgb.T
    wb_gains = np.asarray(isp_params["wb_gains"], dtype=np.float32).reshape(1, 1, 3)
    return (balanced / np.maximum(wb_gains, 1e-6)).astype(np.float32)


def raw_to_clean_energy_field(raw_data, isp_params):
    raw_linear = normalize_raw_mosaic(raw_data, isp_params)
    return demosaic_raw_to_camera_rgb(raw_linear, isp_params["cfa_pattern"])


def fit_ccm_from_linear_srgb(clean_energy_field, reference_linear_srgb, isp_params, max_samples=300000):
    clean_shape = clean_energy_field.shape
    reference_shape = reference_linear_srgb.shape
    if clean_shape[:2] != reference_shape[:2]:
        fit_w = min(clean_shape[1], reference_shape[1])
        fit_h = min(clean_shape[0], reference_shape[0])
        clean_energy_field = center_crop_even(clean_energy_field, fit_w, fit_h)
        reference_linear_srgb = center_crop_even(reference_linear_srgb, fit_w, fit_h)

    wb_gains = np.asarray(isp_params["wb_gains"], dtype=np.float32).reshape(1, 1, 3)
    balanced = clean_energy_field * wb_gains
    x = balanced.reshape(-1, 3)
    y = reference_linear_srgb.reshape(-1, 3)

    mask = (
        np.all(np.isfinite(x), axis=1)
        & np.all(np.isfinite(y), axis=1)
        & (np.max(x, axis=1) > 1e-4)
        & (np.max(x, axis=1) < 0.95)
        & (np.max(y, axis=1) > 1e-4)
        & (np.max(y, axis=1) < 0.95)
    )
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        raise ValueError("No valid pixels for CCM fitting")
    if idx.size > max_samples:
        rng = np.random.default_rng(12345)
        idx = rng.choice(idx, size=max_samples, replace=False)

    a, *_ = np.linalg.lstsq(x[idx], y[idx], rcond=None)
    ccm = validate_ccm_matrix(a.T.astype(np.float32), name="fitted ccm_srgb_from_cam")
    pred = np.clip(balanced @ ccm.T, 0.0, 1.0)
    mae = np.mean(np.abs(pred - reference_linear_srgb), axis=(0, 1))
    return ccm, {
        "fit_pixels": int(idx.size),
        "clean_shape": list(clean_shape),
        "reference_shape": list(reference_shape),
        "fit_shape": list(clean_energy_field.shape),
        "determinant": float(np.linalg.det(ccm)),
        "reference": "rawpy linear sRGB, use_camera_wb=True, gamma=(1,1), no_auto_bright=True",
        "mae_rgb": [float(v) for v in mae],
        "mae_mean": float(np.mean(mae)),
    }


def inverse_isp_to_camera_rgb(srgb, isp_params):
    linear_srgb = srgb_to_linear(srgb)
    ccm = validate_ccm_matrix(isp_params["ccm_srgb_from_cam"])
    cam_from_srgb = np.linalg.inv(ccm)
    cam_rgb = linear_srgb @ cam_from_srgb.T

    wb_gains = np.asarray(isp_params["wb_gains"], dtype=np.float32).reshape(1, 1, 3)
    cam_rgb = cam_rgb / np.maximum(wb_gains, 1e-6)
    return np.clip(cam_rgb, 0.0, 1.0).astype(np.float32)


def validate_cfa_pattern(cfa_pattern):
    pattern = str(cfa_pattern).upper()
    if pattern not in STANDARD_BAYER_PATTERNS:
        raise ValueError(f"cfa_pattern must be one of {sorted(STANDARD_BAYER_PATTERNS)}, got {cfa_pattern!r}")
    return pattern


def qpd_quad_bayer_sample(camera_rgb):
    pattern = QPD_CFA_PATTERN
    channel_index = {"R": 0, "G": 1, "B": 2}
    h, w, _ = camera_rgb.shape
    raw = np.empty((h, w), dtype=np.float32)

    for block_y in range(2):
        for block_x in range(2):
            color = pattern[block_y * 2 + block_x]
            ch = channel_index[color]
            y0 = block_y * 2
            x0 = block_x * 2
            raw[y0::4, x0::4] = camera_rgb[y0::4, x0::4, ch]
            raw[y0 + 1::4, x0::4] = camera_rgb[y0 + 1::4, x0::4, ch]
            raw[y0::4, x0 + 1::4] = camera_rgb[y0::4, x0 + 1::4, ch]
            raw[y0 + 1::4, x0 + 1::4] = camera_rgb[y0 + 1::4, x0 + 1::4, ch]
    return raw


def qpd_quad_bayer_sample_subpixel_2x(qpd_rgb_2x):
    if qpd_rgb_2x.shape[0] % 2 or qpd_rgb_2x.shape[1] % 2:
        raise ValueError("subpixel QPD RGB must have even height and width")
    return qpd_quad_bayer_sample(qpd_rgb_2x)


def center_crop_even(array, width, height):
    if width is None and height is None:
        return array
    if width is None or height is None:
        raise ValueError("--crop must be WIDTHxHEIGHT, for example 3000x2000")

    h, w = array.shape[:2]
    crop_w = min(width, w)
    crop_h = min(height, h)
    crop_w -= crop_w % 2
    crop_h -= crop_h % 2
    x0 = ((w - crop_w) // 2) // 2 * 2
    y0 = ((h - crop_h) // 2) // 2 * 2
    return array[y0:y0 + crop_h, x0:x0 + crop_w, ...]


def parse_crop(crop):
    if crop is None:
        return None, None
    parts = crop.lower().replace("*", "x").split("x")
    if len(parts) != 2:
        raise ValueError("--crop must be WIDTHxHEIGHT, for example 3000x2000")
    width, height = int(parts[0]), int(parts[1])
    if width < 2 or height < 2:
        raise ValueError("--crop dimensions must both be at least 2 pixels")
    return width, height


def apply_qsc_crosstalk(qpd_rgb, qsc_configs):
    h, w, _ = qpd_rgb.shape
    h2 = h - (h % 2)
    w2 = w - (w % 2)
    qpd_rgb = qpd_rgb[:h2, :w2, :]
    sim = QpdSimulate(qsc_configs)
    mask = np.zeros((h2, w2), dtype=np.float32)
    return np.clip(sim(qpd_rgb, mask_fore=mask), 0.0, 1.0)


def load_noise_row(noise_table_path, iso, rng):
    noise_table_path = Path(noise_table_path)
    if noise_table_path.suffix.lower() == ".csv":
        with open(noise_table_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []
            rows = list(reader)
        for row in rows:
            for key, value in list(row.items()):
                try:
                    row[key] = float(value)
                except (TypeError, ValueError):
                    pass
    else:
        try:
            import openpyxl
        except ImportError as exc:
            fallback = noise_table_path.with_suffix(".csv")
            if fallback.exists():
                return load_noise_row(fallback, iso, rng)
            raise ImportError(
                "Reading .xlsx noise tables requires openpyxl. "
                "Install openpyxl or provide a .csv noise table."
            ) from exc

        wb = openpyxl.load_workbook(noise_table_path, data_only=True)
        ws = wb[wb.sheetnames[0]]
        headers = [str(ws.cell(1, c).value).strip() for c in range(1, ws.max_column + 1)]
        rows = []
        for r in range(2, ws.max_row + 1):
            item = {headers[c - 1]: ws.cell(r, c).value for c in range(1, ws.max_column + 1)}
            if item.get("ISO") is not None:
                rows.append(item)
    if not rows:
        raise ValueError(f"No ISO rows found in {noise_table_path}")

    required_columns = {"ISO", "k-10bit", "b-10bit"}
    missing_columns = required_columns.difference(rows[0].keys())
    if missing_columns:
        raise ValueError(f"Noise table {noise_table_path} is missing columns: {sorted(missing_columns)}")

    normalized_rows = []
    for row_index, row in enumerate(rows, start=2):
        out = dict(row)
        try:
            out["ISO"] = float(row["ISO"])
            out["k-10bit"] = float(row["k-10bit"])
            out["b-10bit"] = float(row["b-10bit"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Noise table row {row_index} must have numeric ISO, k-10bit, and b-10bit values"
            ) from exc
        normalized_rows.append(out)

    rows = sorted(normalized_rows, key=lambda row: row["ISO"])
    available_iso = [float(row["ISO"]) for row in rows]

    def random_row(reason):
        idx = int(rng.integers(0, len(rows)))
        out = dict(rows[idx])
        out["requested_iso"] = None if iso is None else float(iso)
        out["noise_iso_selection"] = reason
        out["available_iso_min"] = min(available_iso)
        out["available_iso_max"] = max(available_iso)
        return out

    if iso is None:
        return random_row("missing_iso_random")

    iso = float(iso)
    exact = [row for row in rows if abs(float(row["ISO"]) - iso) < 1e-6]
    if exact:
        out = dict(exact[0])
        out["requested_iso"] = iso
        out["noise_iso_selection"] = "exact_iso_match"
        return out

    if iso < min(available_iso) or iso > max(available_iso):
        return random_row("out_of_range_random")

    return random_row("not_in_table_random")


def add_poisson_gaussian_noise(raw_linear, noise_row, black_level, white_level, bit_depth, rng):
    if bit_depth != 10:
        raise ValueError("The current noise table is fixed to the 10bit DN domain.")
    signal_dn = np.clip(raw_linear, 0.0, 1.0) * (white_level - black_level)

    # Assumption: the table stores 10-bit variance model parameters, variance = k * signal + b.
    k = float(noise_row["k-10bit"])
    b = float(noise_row["b-10bit"])
    variance = np.maximum(k * signal_dn + b, 0.0)
    noisy_signal_dn = signal_dn + rng.normal(0.0, np.sqrt(variance), size=signal_dn.shape)
    return np.clip(noisy_signal_dn / max(white_level - black_level, 1), 0.0, 1.0).astype(np.float32)


def quantize_raw(raw_linear, black_level, white_level, bit_depth):
    full_scale = (1 << bit_depth) - 1
    raw_dn = black_level + np.clip(raw_linear, 0.0, 1.0) * (white_level - black_level)
    return np.clip(np.rint(raw_dn), 0, full_scale).astype(np.uint16)


def save_preview(raw_quantized, black_level, white_level, output_path):
    preview = (raw_quantized.astype(np.float32) - black_level) / max(white_level - black_level, 1)
    preview = np.clip(preview, 0.0, 1.0)
    cv2.imwrite(str(output_path), np.rint(preview * 65535.0).astype(np.uint16))


def save_srgb_image(srgb, output_path):
    srgb = np.nan_to_num(srgb, nan=0.0, posinf=1.0, neginf=0.0)
    rgb_u8 = np.rint(np.clip(srgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    cv2.imwrite(str(output_path), rgb_u8[..., ::-1])


def save_linear_rgb_preview(linear_rgb, output_path):
    preview = np.clip(linear_rgb, 0.0, 1.0)
    save_srgb_image(linear_to_srgb(preview), output_path)


def load_json_or_default(path, default):
    if path is None:
        return dict(default)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    merged = dict(default)
    merged.update(data)
    return merged


def main():
    parser = argparse.ArgumentParser(description="RAW/sRGB ISP round-trip to QPD/QSC 2x2 raw simulator")
    parser.add_argument("--input", required=True, help="Input RAW/DNG or sRGB image path")
    parser.add_argument(
        "--input-kind",
        choices=("auto", "raw", "srgb"),
        default="auto",
        help="raw reads Bayer data and ISP params from RAW metadata; srgb uses --isp-json/default params",
    )
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    parser.add_argument("--isp-json", help="Optional ISP parameter JSON overrides; sRGB mode uses defaults plus this file")
    parser.add_argument("--qsc-json", help="QSC config JSON; missing fields use defaults")
    parser.add_argument("--noise-table", default="noise_table.csv", help="Noise table .xlsx or .csv path")
    parser.add_argument("--iso", type=float, help="Override ISO used for noise table lookup")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--crop", default="3000x2000", help="Center crop before ISP/QSC, default 3000x2000")
    parser.add_argument(
        "--ccm-source",
        choices=("auto", "rawpy-fit", "metadata", "identity"),
        default="auto",
        help="auto fits a reversible 3x3 CCM against rawpy linear sRGB when RAW has a valid color matrix.",
    )
    parser.add_argument(
        "--qpd-readout-mode",
        choices=("same", "subpixel"),
        default="same",
        help="same keeps raw size equal to input; subpixel expands each clean pixel to a 2x2 QPD readout",
    )
    parser.add_argument("--skip-qsc", action="store_true")
    parser.add_argument("--skip-noise", action="store_true")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)

    input_path = Path(args.input)
    raw_suffixes = {
        ".dng",
        ".arw",
        ".cr2",
        ".cr3",
        ".nef",
        ".nrw",
        ".raf",
        ".raw",
        ".rw2",
        ".orf",
        ".pef",
        ".srw",
    }
    input_kind = args.input_kind
    if input_kind == "auto":
        input_kind = "raw" if input_path.suffix.lower() in raw_suffixes else "srgb"

    override_isp_params = None
    if args.isp_json is not None:
        with open(args.isp_json, "r", encoding="utf-8") as f:
            override_isp_params = json.load(f)

    crop_width, crop_height = parse_crop(args.crop)
    linear_srgb = None
    roundtrip_error = None
    ccm_fit_diagnostics = None
    if input_kind == "raw":
        raw_data, isp_params = load_raw_with_isp_params(input_path, override_isp_params)
        raw_data = center_crop_even(raw_data, crop_width, crop_height)
        clean_energy_field = raw_to_clean_energy_field(raw_data, isp_params)
        requested_ccm_source = args.ccm_source
        has_rawpy_color_matrix = bool(isp_params.get("has_rawpy_color_matrix"))
        if requested_ccm_source == "auto":
            requested_ccm_source = "rawpy-fit" if has_rawpy_color_matrix else "identity"

        if requested_ccm_source == "rawpy-fit":
            if not has_rawpy_color_matrix:
                raise ValueError(
                    "Cannot fit CCM because RAW metadata has no rawpy color matrix. "
                    "Use --ccm-source identity, or provide valid ISP parameters with --isp-json."
                )
            reference_linear_srgb = load_rawpy_linear_srgb(input_path, crop_width, crop_height)
            fitted_ccm, ccm_fit_diagnostics = fit_ccm_from_linear_srgb(
                clean_energy_field, reference_linear_srgb, isp_params
            )
            isp_params["ccm_srgb_from_cam"] = fitted_ccm.tolist()
            isp_params["ccm_source"] = "rawpy-fit"
        elif requested_ccm_source == "identity":
            isp_params["ccm_srgb_from_cam"] = np.eye(3, dtype=np.float32).tolist()
            isp_params["ccm_source"] = "no_rawpy_color_matrix_identity" if not has_rawpy_color_matrix else "identity"
        else:
            override_has_ccm = override_isp_params is not None and "ccm_srgb_from_cam" in override_isp_params
            if not override_has_ccm:
                raise ValueError(
                    "--ccm-source metadata now requires a 3x3 --isp-json ccm_srgb_from_cam override. "
                    "Use --ccm-source rawpy-fit or auto for RAW-derived CCM."
                )
            isp_params["ccm_source"] = "provided_isp_json_ccm"
        isp_params["ccm_srgb_from_cam"] = validate_ccm_matrix(isp_params["ccm_srgb_from_cam"]).tolist()
        isp_params["ccm_source_requested"] = args.ccm_source

        linear_srgb = apply_forward_isp_linear(clean_energy_field, isp_params)
        srgb = linear_to_srgb(linear_srgb).astype(np.float32)
        roundtrip_camera_rgb = inverse_linear_isp_to_camera_rgb(linear_srgb, isp_params)
        roundtrip_diff = roundtrip_camera_rgb - clean_energy_field
        roundtrip_error = {
            "max_abs": float(np.max(np.abs(roundtrip_diff))),
            "mean_abs": float(np.mean(np.abs(roundtrip_diff))),
        }
    else:
        isp_params = load_json_or_default(args.isp_json, DEFAULT_ISP_PARAMS)
        isp_params["source_kind"] = "srgb"
        isp_params["ccm_source"] = "provided_srgb_params"
        isp_params["ccm_source_requested"] = args.ccm_source
        isp_params["ccm_srgb_from_cam"] = validate_ccm_matrix(isp_params["ccm_srgb_from_cam"]).tolist()
        srgb = load_srgb_image(input_path)
        srgb = center_crop_even(srgb, crop_width, crop_height)
        clean_energy_field = inverse_isp_to_camera_rgb(srgb, isp_params)
        linear_srgb = apply_forward_isp_linear(clean_energy_field, isp_params)
        roundtrip_error = {
            "max_abs": None,
            "mean_abs": None,
            "note": "sRGB input is not guaranteed reversible because it may already be gamma encoded, clipped, and quantized.",
        }

    qsc_configs = load_json_or_default(args.qsc_json, DEFAULT_QSC_CONFIGS)
    if args.iso is not None:
        isp_params["iso"] = float(args.iso)
        isp_params["iso_source"] = "command_line_override"

    input_bit_depth = int(isp_params["bit_depth"])
    input_black_level = float(isp_params["black_level"])
    input_white_level = float(isp_params["white_level"])
    output_bit_depth = int(QPD_OUTPUT_LEVELS["bit_depth"])
    output_black_level = float(QPD_OUTPUT_LEVELS["black_level"])
    output_white_level = float(QPD_OUTPUT_LEVELS["white_level"])
    iso = None if isp_params.get("iso") is None else float(isp_params["iso"])

    qpd_rgb = np.clip(clean_energy_field, 0.0, 1.0)
    qpd_readout_scale = 1
    if args.qpd_readout_mode == "subpixel":
        qpd_rgb = np.repeat(np.repeat(qpd_rgb, 2, axis=0), 2, axis=1)
        qpd_readout_scale = 2

    if not args.skip_qsc:
        qpd_rgb = apply_qsc_crosstalk(qpd_rgb, qsc_configs)

    if args.qpd_readout_mode == "same":
        raw_linear = qpd_quad_bayer_sample(qpd_rgb)
    else:
        raw_linear = qpd_quad_bayer_sample_subpixel_2x(qpd_rgb)

    noise_row = None
    if not args.skip_noise:
        noise_row = load_noise_row(args.noise_table, iso, rng)
        raw_linear = add_poisson_gaussian_noise(
            raw_linear,
            noise_row,
            output_black_level,
            output_white_level,
            output_bit_depth,
            rng,
        )

    raw_quantized = quantize_raw(raw_linear, output_black_level, output_white_level, output_bit_depth)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "clean_energy_field.npy", clean_energy_field.astype(np.float32))
    np.save(output_dir / "isp_linear_srgb.npy", linear_srgb.astype(np.float32))
    np.save(output_dir / "qpd_raw.npy", raw_quantized)
    save_srgb_image(srgb, output_dir / "isp_srgb.png")
    save_linear_rgb_preview(clean_energy_field, output_dir / "clean_energy_preview.png")
    save_preview(raw_quantized, output_black_level, output_white_level, output_dir / "qpd_raw_preview.png")

    metadata = {
        "input": str(args.input),
        "input_kind": input_kind,
        "crop": args.crop,
        "isp_srgb_source": "linear_to_srgb(clip(isp_linear_srgb)) reversible ISP preview",
        "isp_srgb": str(output_dir / "isp_srgb.png"),
        "clean_energy_field": str(output_dir / "clean_energy_field.npy"),
        "isp_linear_srgb": str(output_dir / "isp_linear_srgb.npy"),
        "clean_energy_domain": "linear camera RGB after black-level subtraction, white-level normalization, and demosaic",
        "reversible_isp_contract": "clean_energy_field -> AWB -> CCM -> isp_linear_srgb; inverse uses inverse(CCM) and inverse AWB on float tensors before gamma/clip/quantization",
        "reversible_isp_roundtrip_error": roundtrip_error,
        "ccm_fit_diagnostics": ccm_fit_diagnostics,
        "shape": list(raw_quantized.shape),
        "qpd_raw_shape": list(raw_quantized.shape),
        "clean_energy_shape": list(clean_energy_field.shape),
        "isp_linear_srgb_shape": list(linear_srgb.shape),
        "dtype": str(raw_quantized.dtype),
        "input_raw_levels": {
            "bit_depth": input_bit_depth,
            "black_level": input_black_level,
            "white_level": input_white_level,
        },
        "qpd_output_levels": {
            "bit_depth": output_bit_depth,
            "black_level": output_black_level,
            "white_level": output_white_level,
        },
        "qpd_readout_mode": args.qpd_readout_mode,
        "qpd_readout_scale": qpd_readout_scale,
        "qpd_cfa_pattern": QPD_CFA_PATTERN,
        "qpd_cfa_layout": QPD_CFA_LAYOUT,
        "isp_params": isp_params,
        "qsc_configs": qsc_configs if not args.skip_qsc else None,
        "noise_row": noise_row,
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"saved {output_dir / 'qpd_raw.npy'}")
    print(f"saved {output_dir / 'qpd_raw_preview.png'}")
    print(f"saved {output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
