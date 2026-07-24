# QPD HWK Raw Simulation Pipeline

This repository builds paired training data for:

```text
QPD raw -> clean energy field image
```

The main target is joint demosaic / denoise training. The clean target is a linear camera RGB energy field, and the input is simulated 2x2 QPD raw with measured full-field HWK response, residual domain randomization, Poisson-Gaussian noise, and 10bit quantization.

## Files

```text
qpd_qsc_pipeline.py      Single-image RAW/DNG -> QPD dataset sample
qpd_hwk_simulator.py    Full-field R/Gr/Gb/B HWK and residual-mix simulator
batch_fivek_pipeline.py Batch download/process MIT-Adobe FiveK DNG files
split_qpd_dataset.py    Create train/val/test manifests for QPD raw -> clean energy
visualize_linear_srgb.py Render isp_linear_srgb.npy as an sRGB PNG
noise_table.csv         10bit noise model table
noise_table.xlsx        Original noise table
```

Generated data directories such as `data/`, `outputs/`, and `dataset/` are intentionally not required to exist before running.

## Requirements

Install the runtime dependencies:

```powershell
python -m pip install rawpy opencv-python numpy openpyxl exifread
```

`openpyxl` is only needed when reading `noise_table.xlsx`. The default examples use `noise_table.csv`.

## Single Image

Run one RAW/DNG image:

```powershell
python qpd_qsc_pipeline.py --input path\to\image.dng --input-kind raw --hwk-dir path\to\qpd_hwk_statistics_4c --output-dir outputs\sample --noise-table noise_table.csv
```

Default settings:

```text
crop: 3000x2000
QPD CFA layout: fixed Quad Bayer RGGB, 2x2 same-color blocks
QPD output size: same as clean target
QPD raw output: 10bit, black=64, white=1023
noise model: k-10bit / b-10bit from noise_table.csv
```

Expected HWK directory layout:

```text
qpd_hwk_statistics_4c/
  qpd_hwk_field_manifest.csv
  field_data/
    1.0m_F1.4_hwk_full_field.csv
    ...
```

The selected full HWK field is center-cropped to the required QPD grid. For a `3000x2000` output, the CFA planes are `(1000,1500,4)` and the HWK field is `(500,750,4)`. The calibration field must be at least this large; HWK fields are not resized.

Select a calibrated condition explicitly when needed:

```powershell
python qpd_qsc_pipeline.py --input path\to\image.dng --input-kind raw --hwk-dir path\to\qpd_hwk_statistics_4c --hwk-distance 1.0m --hwk-aperture F1.4 --output-dir outputs\sample
```

Use `--hwk-config path\to\hwk_config.json` to override jitter and residual-mix settings. `--skip-qpd-sim` is a debug mode that retains Quad RGGB sampling but bypasses HWK/RDM.

The first CSV load writes an adjacent `.csv.npz` binary cache. Later batch subprocesses load this cache instead of reparsing the full CSV. Use `--no-hwk-cache` when the calibration directory must remain read-only.

The simulated QPD raw always uses quad Bayer RGGB layout. The input camera CFA is only used when reading and demosaicing the source RAW into `clean_energy_field`.

Per-sample outputs:

```text
clean_energy_field.npy   Target. Linear camera RGB after black/white normalization and demosaic.
qpd_raw.npy              Input. Simulated QPD raw, uint16, 10bit levels.
isp_linear_srgb.npy      Optional reversible linear ISP intermediate (`--save-isp-linear`).
isp_srgb.png             Preview of the reversible ISP result.
clean_energy_preview.png Preview of the clean energy field.
qpd_raw_preview.png      Preview of qpd_raw.
metadata.json            Parameters, levels, ISO/noise row, CCM info, roundtrip error.
```

`clean_energy_field.npy` is stored as `float16` by default while all ISP and simulation calculations remain `float32`. Use `--clean-dtype float32` when full float32 target storage is required. The optional `isp_linear_srgb.npy` is only written with `--save-isp-linear`.

Render the stored linear-sRGB tensor as an 8bit sRGB PNG:

```powershell
python qpd_qsc_pipeline.py --input path\to\image.dng --input-kind raw --hwk-dir path\to\qpd_hwk_statistics_4c --output-dir outputs\sample --save-isp-linear
python visualize_linear_srgb.py --input-npy outputs\sample\isp_linear_srgb.npy --metadata-json outputs\sample\metadata.json
```

The script validates the NPY shape against the supplied metadata, applies only the sRGB transfer function, and writes `reconstructed_srgb.png` beside the metadata. AWB and CCM are not applied again because they are already included in `isp_linear_srgb.npy`. Use `--bit-depth 16` for a 16bit PNG or `--output path\to\preview.png` to choose the output path.

## FiveK Batch Processing

Download and process the full MIT-Adobe FiveK DNG set:

```powershell
python batch_fivek_pipeline.py --download-all --raw-dir data\raw_samples\fivek_full --output-root outputs\fivek_full --hwk-dir path\to\qpd_hwk_statistics_4c --noise-table noise_table.csv
```

Only download DNG files, without processing:

```powershell
python batch_fivek_pipeline.py --download-all --download-only --raw-dir data\raw_samples\fivek_full --output-root outputs\fivek_full
```

Test the first 10 files:

```powershell
python batch_fivek_pipeline.py --download-all --limit 10 --raw-dir data\raw_samples\fivek_full --output-root outputs\fivek_full --hwk-dir path\to\qpd_hwk_statistics_4c --noise-table noise_table.csv
```

Resume processing an existing DNG directory:

```powershell
python batch_fivek_pipeline.py --raw-dir data\raw_samples\fivek_full --output-root outputs\fivek_full --hwk-dir path\to\qpd_hwk_statistics_4c --noise-table noise_table.csv
```

By default, existing completed samples are skipped. A completed sample has:

```text
metadata.json
qpd_raw.npy
clean_energy_field.npy
```

Its metadata must also match the crop, fixed Quad RGGB layout, HWK simulator version, HWK path/config hash, and requested distance/aperture. Legacy QSC outputs are reprocessed.

Batch mode defaults to compact training outputs: `clean_energy_field.npy` uses `float16`, `isp_linear_srgb.npy` is omitted, and preview PNGs are omitted. Add `--clean-dtype float32`, `--save-isp-linear`, or `--save-previews` only when those debugging artifacts are needed.

Before launching the single-image pipeline, the batch script checks the visible RAW dimensions against `--crop` (default `3000x2000`). An undersized image is marked as `skipped_small_image`, does not get a per-sample output directory, and is recorded only in `batch_summary.json` with its source and requested dimensions. `skipped_small_image_count` reports the total separately from successful and failed samples.

Force reprocessing:

```powershell
python batch_fivek_pipeline.py --raw-dir data\raw_samples\fivek_full --output-root outputs\fivek_full --hwk-dir path\to\qpd_hwk_statistics_4c --noise-table noise_table.csv --no-skip-existing
```

Some FiveK files are not standard 2x2 Bayer CFA. These are recorded as failures and skipped, unless `--fail-fast` is set.

Batch processing accepts the same HWK condition/config options:

```powershell
python batch_fivek_pipeline.py --raw-dir data\raw_samples\fivek_full --output-root outputs\fivek_full --hwk-dir path\to\qpd_hwk_statistics_4c --hwk-distance 1.0m --hwk-aperture F1.4 --hwk-config path\to\hwk_config.json
```

## Dataset Split

Create train/val/test manifests:

```powershell
python split_qpd_dataset.py --source-root outputs\fivek_full --output-root dataset\qpd_fivek --train 0.8 --val 0.1 --test 0.1 --seed 2026
```

By default, all valid samples participate in the split. Use `--num-samples` to select a reproducible random subset before applying the train/val/test ratios:

```powershell
python split_qpd_dataset.py --source-root outputs\fivek_full --output-root dataset\qpd_fivek_1000 --num-samples 1000 --seed 2026
```

The requested count must be a positive integer no larger than the number of valid samples. `split_summary.json` records both `total_valid_samples` and `participating_samples`.

To preserve sorted sample-directory order, select the first `N` valid samples and split them into contiguous train/val/test ranges:

```powershell
python split_qpd_dataset.py --source-root outputs\fivek_full --output-root dataset\qpd_fivek_seq400 --num-samples 400 --split-order sequential --train 0.8 --val 0.1 --test 0.1
```

With this example, the first 320 sorted samples become train, the next 40 become val, and the final 40 become test. `--seed` only affects the default `random` mode. The selected mode is recorded as `split_order` in `split_summary.json`.

This writes:

```text
train.csv / train.json
val.csv / val.json
test.csv / test.json
all.csv / all.json
split_summary.json
```

Each row maps:

```text
input_qpd_raw -> target_clean_energy
```

The manifests include `qpd_shape`, `target_shape`, `target_dtype`, simulator type, and selected HWK condition. Only current HWK-simulator outputs are included.

By default, only manifests are written. To materialize files into split folders:

```powershell
python split_qpd_dataset.py --source-root outputs\fivek_full --output-root dataset\qpd_fivek --materialize hardlink
```

Use `--materialize copy` only if you intentionally want to duplicate the large `.npy` files.

## ISP And CCM Notes

The reversible ISP contract is:

```text
clean_energy_field -> AWB -> CCM -> isp_linear_srgb
isp_linear_srgb -> inverse CCM -> inverse AWB -> clean_energy_field
```

`isp_srgb.png` is only a preview created from the reversible linear ISP tensor. It is not used as training data.

CCM selection:

```text
metadata (default):
  use raw.color_matrix[:, :3] as camera-to-sRGB CCM
  if unavailable, derive the CCM from raw.rgb_xyz_matrix through a normalized pseudo-inverse
rawpy-fit:
  fit a reversible 3x3 CCM against rawpy linear sRGB
identity:
  bypass CCM and keep only AWB
```

The default `metadata` path uses LibRaw/rawpy's camera-to-sRGB `color_matrix` when available. If `color_matrix` is unavailable, it derives the same style of CCM from `rgb_xyz_matrix` with `pinv(row_normalize(raw.rgb_xyz_matrix[:3, :3] @ srgb_to_xyz_d65))`. Use `--ccm-source rawpy-fit` when you want the preview to match rawpy's full postprocess output more closely despite demosaic/scale differences.

The roundtrip error is written into `metadata.json`.

## Noise And Quantization

The output QPD raw is fixed to:

```text
10bit
black level = 64
white level = 1023
```

Noise uses the 10bit domain directly:

```text
variance = k-10bit * signal_dn + b-10bit
```

ISO handling:

```text
If ISO is available and exactly exists in noise_table.csv -> use that row.
If ISO is missing, out of range, or not in the table -> randomly choose one noise row.
```

The chosen noise row is recorded in `metadata.json`.

## Storage Warning

The full FiveK DNG archive is about 50GB. Default batch output for a `3000x2000` sample stores approximately 45.8 MiB:

```text
clean_energy_field.npy  float16, about 34.3 MiB
qpd_raw.npy             uint16, about 11.4 MiB
metadata.json           a few KiB
```

Optional `isp_linear_srgb.npy` adds about 68.7 MiB, and the three preview PNGs commonly add tens of MiB. Even compact output can exceed 200GB for all 5000 images, so check available storage before a full run.
