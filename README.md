# QPD/QSC Raw Simulation Pipeline

This repository builds paired training data for:

```text
QPD raw -> clean energy field image
```

The main target is joint demosaic / denoise training. The clean target is a linear camera RGB energy field, and the input is simulated 2x2 QPD raw with QSC perturbation, Poisson-Gaussian noise, and 10bit quantization.

## Files

```text
qpd_qsc_pipeline.py      Single-image RAW/DNG -> QPD dataset sample
batch_fivek_pipeline.py Batch download/process MIT-Adobe FiveK DNG files
split_qpd_dataset.py    Create train/val/test manifests for QPD raw -> clean energy
crosstalk_sim.py        QPD/QSC 2x2 perturbation simulator
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
python qpd_qsc_pipeline.py --input path\to\image.dng --input-kind raw --output-dir outputs\sample --noise-table noise_table.csv
```

Default settings:

```text
crop: 3000x2000
QPD CFA layout: fixed quad Bayer RGGB, 2x2 same-color blocks
QPD raw output: 10bit, black=64, white=1023
noise model: k-10bit / b-10bit from noise_table.csv
```

Optional QPD readout modes:

```powershell
python qpd_qsc_pipeline.py --input path\to\image.dng --input-kind raw --qpd-readout-mode same
python qpd_qsc_pipeline.py --input path\to\image.dng --input-kind raw --qpd-readout-mode subpixel
```

`same` keeps the QPD raw size equal to the clean target. `subpixel` expands each clean pixel into a 2x2 QPD readout grid; QSC perturbation is applied on that expanded grid before CFA sampling.

The simulated QPD raw always uses quad Bayer RGGB layout. The input camera CFA is only used when reading and demosaicing the source RAW into `clean_energy_field`.

Per-sample outputs:

```text
clean_energy_field.npy   Target. Linear camera RGB after black/white normalization and demosaic.
qpd_raw.npy              Input. Simulated QPD raw, uint16, 10bit levels.
isp_linear_srgb.npy      Reversible linear ISP intermediate.
isp_srgb.png             Preview of the reversible ISP result.
clean_energy_preview.png Preview of the clean energy field.
qpd_raw_preview.png      Preview of qpd_raw.
metadata.json            Parameters, levels, ISO/noise row, CCM info, roundtrip error.
```

## FiveK Batch Processing

Download and process the full MIT-Adobe FiveK DNG set:

```powershell
python batch_fivek_pipeline.py --download-all --raw-dir data\raw_samples\fivek_full --output-root outputs\fivek_full --noise-table noise_table.csv
```

Only download DNG files, without processing:

```powershell
python batch_fivek_pipeline.py --download-all --download-only --raw-dir data\raw_samples\fivek_full --output-root outputs\fivek_full
```

Test the first 10 files:

```powershell
python batch_fivek_pipeline.py --download-all --limit 10 --raw-dir data\raw_samples\fivek_full --output-root outputs\fivek_full --noise-table noise_table.csv
```

Resume processing an existing DNG directory:

```powershell
python batch_fivek_pipeline.py --raw-dir data\raw_samples\fivek_full --output-root outputs\fivek_full --noise-table noise_table.csv
```

By default, existing completed samples are skipped. A completed sample has:

```text
metadata.json
qpd_raw.npy
clean_energy_field.npy
```

Its metadata must also match the requested crop/readout settings and the fixed quad Bayer RGGB layout; older outputs with a different QPD CFA contract are reprocessed.

Force reprocessing:

```powershell
python batch_fivek_pipeline.py --raw-dir data\raw_samples\fivek_full --output-root outputs\fivek_full --noise-table noise_table.csv --no-skip-existing
```

Some FiveK files are not standard 2x2 Bayer CFA. These are recorded as failures and skipped, unless `--fail-fast` is set.

Batch processing also accepts the single-image QPD mode options:

```powershell
python batch_fivek_pipeline.py --raw-dir data\raw_samples\fivek_full --output-root outputs\fivek_full --qpd-readout-mode subpixel
```

## Dataset Split

Create train/val/test manifests:

```powershell
python split_qpd_dataset.py --source-root outputs\fivek_full --output-root dataset\qpd_fivek --train 0.8 --val 0.1 --test 0.1 --seed 2026
```

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

The manifests include both `qpd_shape` and `target_shape`, so size-changing modes such as `--qpd-readout-mode subpixel` are explicit.

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
auto:
  rawpy color matrix exists -> fit a reversible 3x3 CCM against rawpy linear sRGB
  no rawpy color matrix -> identity CCM
```

The default `auto` path uses `rawpy-fit` whenever `rawpy.rgb_xyz_matrix` exists. It does not inspect the matrix shape or effective channel count. RAW metadata initializes `ccm_srgb_from_cam` from the default ISP params; the fitted CCM overwrites it before preview/output generation. `--ccm-source metadata` is only for explicit debugging with a 3x3 `ccm_srgb_from_cam` supplied through `--isp-json`.

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

The full FiveK DNG archive is about 50GB. The generated outputs are much larger because each sample stores:

```text
clean_energy_field.npy
isp_linear_srgb.npy
qpd_raw.npy
preview PNGs
metadata
```

Plan for hundreds of GB if processing the full dataset.
