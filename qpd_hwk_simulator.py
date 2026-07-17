#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QPD HWK Simulator V2
====================

功能
----
1. 输入是 linear camera RGB，不执行 inverse gamma、CCM 或 AWB。
2. 对 (H,W,3) linear camera RGB 执行 Quad RGGB sampling，得到 (H/2,W/2,4)：
   R / Gr / Gb / B。
3. 从 field_data CSV 读取完整视场四通道 h/w/kappa 参数场。
4. 在每个 CFA 平面内部按 2x2 QPD 单元执行 h/w/kappa 重分配。
5. 可选为每个 QPD 单元独立生成 (4,4,4) residual mix：
   输入PD x 输出PD x CFA通道；混合强度由中心向视场边缘平滑增强。
6. 输出四 CFA 平面，也可恢复为单通道 RGGB RAW mosaic。

尺寸关系
--------
linear RGB   : (3072, 4096, 3)
CFA planes   : (1536, 2048, 4)
HWK field    : ( 768, 1024, 4)
RAW mosaic   : (3072, 4096)

最小用法
--------
    import numpy as np

    from qpd_hwk_simulator import (
        HwkBank,
        QpdHwkSimulator,
        quad_rggb_planes_to_mosaic,
    )

    # field_data CSV 统计结果根目录：
    # qpd_hwk_statistics_4c/
    # ├── qpd_hwk_field_manifest.csv
    # └── field_data/
    #     ├── 1.0m_F1.4_hwk_full_field.csv
    #     └── ...
    bank = HwkBank("./qpd_hwk_statistics_4c")

    config = {
        # 只会从 HwkBank 中实际存在的物距/光圈组合中选择。
        "distance_candidates": ["1.0m", "1.5m", "2.0m"],
        "aperture_candidates": ["F1.4", "F2.0", "F4.0"],

        # 无效标定点回退为理想 QPD：h=1, w=1, kappa=0。
        "invalid_policy": "ideal",

        # 对实测 HWK 参数施加微小训练扰动。
        # channel：R/Gr/Gb/B 每个通道共享一组小偏移，
        # 不破坏实测全视场参数场的空间结构。
        "prob_hwk_jitter": 1.0,
        "hwk_jitter_scope": "channel",
        "hwk_jitter_distribution": "normal",
        "hwk_jitter_std": {
            "h": (0.0, 0.005),
            "w": (0.0, 0.005),
            "kappa": (0.0, 0.002),
        },

        # 可选 residual mix。每个 QPD 独立生成矩阵，强度中心弱、边缘强。
        "prob_rdm_mix": 0.3,
        "rdm_mix_strength": (0.0, 0.05),
        "rdm_mix_center_ratio": (0.2, 0.5),
        "rdm_mix_field_power": (1.0, 2.0),
        "rdm_mix_channel_mode": "independent",  # 或 "shared"
        "prob_flatten": 0.8,

        # 默认不裁剪，便于检查 QPD 块内能量守恒。
        "output_clip": None,
    }

    simulator = QpdHwkSimulator(bank, config=config, seed=123)

    # 输入必须已经是 linear camera RGB。
    # src_linear_rgb.shape == (3072, 4096, 3)
    src_linear_rgb = np.load("linear_rgb.npy").astype(np.float32)

    dst_cfa, meta = simulator.simulate_from_camera_rgb(
        src_linear_rgb,
        distance="1.0m",   # None：按 config 候选条件随机选择
        aperture="F1.4",   # None：按 config 候选条件随机选择
        return_meta=True,
    )

    # dst_cfa.shape == (1536, 2048, 4)，通道顺序 R/Gr/Gb/B。
    dst_raw = quad_rggb_planes_to_mosaic(dst_cfa)
    # dst_raw.shape == (3072, 4096)

    print("available conditions:", bank.available_conditions)
    print("selected condition:", meta["distance"], meta["aperture"])
    print("output CFA shape:", dst_cfa.shape)
    print("output RAW shape:", dst_raw.shape)
    print("residual mix applied:", meta["rdm_mix"]["applied"])
    print(
        "max QPD energy error:",
        meta["max_qpd_energy_error_before_output_clip"],
    )

如果已经完成 CFA sampling
-------------------------
    from qpd_hwk_simulator import quad_rggb_sample_to_planes

    src_cfa = quad_rggb_sample_to_planes(src_linear_rgb)
    dst_cfa, meta = simulator(
        src_cfa,
        distance="1.0m",
        aperture="F1.4",
        return_meta=True,
    )

关闭所有随机增强
----------------
    deterministic_config = {
        "invalid_policy": "ideal",
        "prob_hwk_jitter": 0.0,
        "prob_rdm_mix": 0.0,
        "output_clip": None,
    }

说明
----
- V2 使用完整视场一一对应，因此没有 field_sampling 和 field_origin。
- 不融合 Gr/Gb；四个 CFA 通道分别读取和应用 h/w/kappa。
- residual mix 的 shape 固定为 (4,4,4)，不是 (4,4,3)。
- residual mix 的 flatten/random 模式按样本选择，具体矩阵按 QPD 单元独立生成。
- residual mix 强度按径向视场连续变化，默认中心弱、边缘强。
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np


CFA_COLORS = ("R", "Gr", "Gb", "B")
PD_ORDER = ("TL", "TR", "BL", "BR")
ConditionKey = Tuple[str, str]
ScalarOrRange = Union[float, int, Sequence[float]]


# ============================================================
# 1. CFA sampling / restoration
# ============================================================
def quad_rggb_sample_to_planes(camera_rgb: np.ndarray) -> np.ndarray:
    """
    对线性 camera RGB 执行 2x2 Quad RGGB sampling，并解包为四个 CFA 平面。

    Parameters
    ----------
    camera_rgb : ndarray, shape (H, W, 3)
        线性 camera RGB；本函数不做 AWB、CCM 或 gamma。

    Returns
    -------
    planes : ndarray, shape (H/2, W/2, 4)
        通道顺序固定为 R, Gr, Gb, B。
    """
    rgb = np.asarray(camera_rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"camera_rgb 应为 (H,W,3)，实际为 {rgb.shape}")
    if rgb.shape[0] % 4 != 0 or rgb.shape[1] % 4 != 0:
        raise ValueError("Quad RGGB sampling 要求高度和宽度均为 4 的倍数")
    if not np.all(np.isfinite(rgb)):
        raise ValueError("camera_rgb 包含 NaN 或 Inf")

    height, width = rgb.shape[:2]
    qpd_height, qpd_width = height // 4, width // 4
    rgb_blocks = rgb.reshape(qpd_height, 4, qpd_width, 4, 3)
    planes = []
    for block_y, block_x, channel in ((0, 0, 0), (0, 2, 1), (2, 0, 1), (2, 2, 2)):
        plane = rgb_blocks[
            :, block_y:block_y + 2, :, block_x:block_x + 2, channel
        ].reshape(height // 2, width // 2)
        planes.append(plane)
    return np.stack(planes, axis=-1)


def quad_rggb_planes_to_mosaic(cfa_planes: np.ndarray) -> np.ndarray:
    """将 (H/2,W/2,4) 的 QPD 平面放回单通道 2x2 Quad RGGB raw。"""
    planes = np.asarray(cfa_planes)
    if planes.ndim != 3 or planes.shape[2] != 4:
        raise ValueError(f"cfa_planes 应为 (Hc,Wc,4)，实际为 {planes.shape}")

    hc, wc, _ = planes.shape
    if hc % 2 != 0 or wc % 2 != 0:
        raise ValueError("Quad RGGB CFA 平面的高度和宽度必须为偶数")
    qpd_height, qpd_width = hc // 2, wc // 2
    raw_blocks = np.empty((qpd_height, 4, qpd_width, 4), dtype=planes.dtype)
    for channel, block_y, block_x in ((0, 0, 0), (1, 0, 2), (2, 2, 0), (3, 2, 2)):
        raw_blocks[:, block_y:block_y + 2, :, block_x:block_x + 2] = (
            planes[..., channel].reshape(qpd_height, 2, qpd_width, 2)
        )
    return raw_blocks.reshape(hc * 2, wc * 2)


# ============================================================
# 2. 标定参数库
# ============================================================
@dataclass(frozen=True)
class HwkEntry:
    """一个物距/光圈条件下的完整四通道 h/w/kappa 参数场。"""

    distance: str
    aperture: str
    source_path: Path

    # shape = (Hq, Wq, 4)，顺序 R/Gr/Gb/B
    h: np.ndarray
    w: np.ndarray
    kappa: np.ndarray
    valid: np.ndarray

    @property
    def key(self) -> ConditionKey:
        return self.distance, self.aperture

    @property
    def spatial_shape(self) -> Tuple[int, int]:
        return int(self.h.shape[0]), int(self.h.shape[1])


class HwkBank:
    """
    从统计程序导出的 full-field CSV 构建：

        hwk_bank[(distance, aperture)] = HwkEntry(...)

    推荐输入目录结构：

        qpd_hwk_statistics/
        ├── qpd_hwk_field_manifest.csv
        └── field_data/
            ├── 1.0m_F1.4_hwk_full_field.csv
            ├── 1.0m_F2.0_hwk_full_field.csv
            └── ...

    CSV 必须包含 R/Gr/Gb/B 四通道的 valid/h/w/kappa 列。旧的 R/G/B
    三通道 CSV 无法恢复 Gr/Gb，因此不会自动复制 G 通道。
    """

    FIELD_SUFFIX = "_hwk_full_field.csv"
    MANIFEST_NAME = "qpd_hwk_field_manifest.csv"
    CACHE_VERSION = 1

    def __init__(
        self,
        source: Union[str, Path, Sequence[Union[str, Path]]],
        *,
        manifest_path: Optional[Union[str, Path]] = None,
        csv_chunk_rows: int = 100_000,
        use_cache: bool = True,
    ) -> None:
        if int(csv_chunk_rows) <= 0:
            raise ValueError("csv_chunk_rows 必须为正整数")

        self.csv_chunk_rows = int(csv_chunk_rows)
        self.use_cache = bool(use_cache)
        self._entries: Dict[ConditionKey, HwkEntry] = {}

        paths = self._discover_paths(source)
        if not paths:
            raise FileNotFoundError(
                "没有找到 *_hwk_full_field.csv；请传入 field_data 目录、"
                "统计结果根目录或具体 CSV 文件"
            )

        manifest_records = self._load_manifest_records(paths, manifest_path)

        for path in paths:
            manifest_row = self._match_manifest_record(path, manifest_records)
            entry = self._load_entry(path, manifest_row)
            if entry.key in self._entries:
                previous = self._entries[entry.key].source_path
                raise ValueError(
                    "同一物距/光圈存在多个 field_data CSV：\n"
                    f"  condition={entry.key}\n"
                    f"  first={previous}\n"
                    f"  second={entry.source_path}"
                )
            self._entries[entry.key] = entry

    @classmethod
    def _discover_paths(
        cls,
        source: Union[str, Path, Sequence[Union[str, Path]]],
    ) -> List[Path]:
        sources = [source] if isinstance(source, (str, Path)) else list(source)
        result: List[Path] = []

        for item in sources:
            path = Path(item)
            if path.is_file():
                if path.suffix.lower() != ".csv":
                    raise ValueError(f"HWK 标定文件必须为 CSV：{path}")
                if not path.name.endswith(cls.FIELD_SUFFIX):
                    raise ValueError(
                        f"CSV 文件名应以 {cls.FIELD_SUFFIX} 结尾：{path}"
                    )
                result.append(path)
                continue

            if not path.is_dir():
                raise FileNotFoundError(f"HWK field_data 路径不存在：{path}")

            # 传入统计结果根目录时，优先只扫描其 field_data 子目录，
            # 避免误读 summary/histogram/correlation 等 CSV。
            field_dir = path / "field_data"
            search_root = field_dir if field_dir.is_dir() else path
            result.extend(sorted(search_root.rglob(f"*{cls.FIELD_SUFFIX}")))

        unique: List[Path] = []
        seen = set()
        for path in result:
            resolved = path.resolve()
            if resolved not in seen:
                unique.append(path)
                seen.add(resolved)
        return unique

    @classmethod
    def _candidate_manifest_paths(
        cls,
        csv_paths: Sequence[Path],
        explicit_manifest: Optional[Union[str, Path]],
    ) -> List[Path]:
        if explicit_manifest is not None:
            path = Path(explicit_manifest)
            if not path.is_file():
                raise FileNotFoundError(f"manifest 不存在：{path}")
            return [path]

        candidates: List[Path] = []
        for csv_path in csv_paths:
            candidates.extend(
                [
                    csv_path.parent / cls.MANIFEST_NAME,
                    csv_path.parent.parent / cls.MANIFEST_NAME,
                ]
            )

        unique: List[Path] = []
        seen = set()
        for path in candidates:
            resolved = path.resolve()
            if path.is_file() and resolved not in seen:
                unique.append(path)
                seen.add(resolved)
        return unique

    @classmethod
    def _load_manifest_records(
        cls,
        csv_paths: Sequence[Path],
        explicit_manifest: Optional[Union[str, Path]],
    ) -> List[Dict[str, str]]:
        records: List[Dict[str, str]] = []
        for manifest in cls._candidate_manifest_paths(csv_paths, explicit_manifest):
            with manifest.open("r", encoding="utf-8-sig", newline="") as file:
                reader = csv.DictReader(file)
                if reader.fieldnames is None:
                    continue
                for row in reader:
                    record = {str(k): str(v) for k, v in row.items() if k is not None}
                    record["__manifest_path__"] = str(manifest)
                    records.append(record)
        return records

    @classmethod
    def _match_manifest_record(
        cls,
        path: Path,
        records: Sequence[Mapping[str, str]],
    ) -> Optional[Mapping[str, str]]:
        if not records:
            return None

        label = path.name[: -len(cls.FIELD_SUFFIX)]
        exact_path_matches: List[Mapping[str, str]] = []
        basename_matches: List[Mapping[str, str]] = []
        label_matches: List[Mapping[str, str]] = []

        resolved = path.resolve()
        for record in records:
            field_csv = str(record.get("field_csv", "")).strip()
            if field_csv:
                manifest_path = Path(record.get("__manifest_path__", "."))
                candidate = Path(field_csv)
                if not candidate.is_absolute():
                    candidate = manifest_path.parent / candidate
                try:
                    if candidate.resolve() == resolved:
                        exact_path_matches.append(record)
                        continue
                except OSError:
                    pass
                if candidate.name == path.name:
                    basename_matches.append(record)

            if str(record.get("scope_label", "")).strip() == label:
                label_matches.append(record)

        for matches in (exact_path_matches, basename_matches, label_matches):
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                raise ValueError(
                    f"manifest 中有多个记录匹配 {path}；请用 manifest_path 指定唯一清单"
                )
        return None

    @classmethod
    def _parse_condition_from_filename(cls, path: Path) -> Tuple[str, str]:
        name = path.name
        if not name.endswith(cls.FIELD_SUFFIX):
            raise ValueError(f"无法识别 field_data 文件名：{path}")
        label = name[: -len(cls.FIELD_SUFFIX)]

        # 最后一段 F... 作为光圈，前面的完整字符串作为物距。
        match = re.match(r"^(?P<distance>.+)_(?P<aperture>F[^_]+)$", label)
        if match is None:
            raise ValueError(
                f"无法从文件名解析物距/光圈：{path.name}。"
                "建议同时提供 qpd_hwk_field_manifest.csv"
            )
        distance = match.group("distance")
        aperture = match.group("aperture").replace("p", ".")
        return distance, aperture

    @classmethod
    def _read_metadata(
        cls,
        path: Path,
        manifest_row: Optional[Mapping[str, str]],
    ) -> Tuple[str, str, Optional[int], Optional[int], str]:
        filename_distance, filename_aperture = cls._parse_condition_from_filename(path)
        if manifest_row is None:
            return filename_distance, filename_aperture, None, None, "unknown"

        distance = str(manifest_row.get("distance", "")).strip() or filename_distance
        aperture = (
            str(manifest_row.get("aperture", "")).strip() or filename_aperture
        ).replace("p", ".")

        def parse_positive_int(name: str) -> Optional[int]:
            text = str(manifest_row.get(name, "")).strip()
            if not text:
                return None
            value = int(float(text))
            if value <= 0:
                raise ValueError(f"{path}: manifest 中 {name} 必须为正数")
            return value

        height = parse_positive_int("qpd_height")
        width = parse_positive_int("qpd_width")
        csv_mode = str(manifest_row.get("csv_mode", "unknown")).strip() or "unknown"
        return distance, aperture, height, width, csv_mode

    @staticmethod
    def _read_header(path: Path) -> List[str]:
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.reader(file)
            try:
                header = next(reader)
            except StopIteration as exc:
                raise ValueError(f"CSV 为空：{path}") from exc
        return [str(x).strip() for x in header]

    @classmethod
    def _validate_header(cls, path: Path, header: Sequence[str]) -> Dict[str, int]:
        index = {name: i for i, name in enumerate(header)}
        if len(index) != len(header):
            raise ValueError(f"{path}: CSV 存在重复列名")

        required = ["qpd_row", "qpd_col"]
        for color in CFA_COLORS:
            required.extend(
                [
                    f"valid_{color}",
                    f"h_{color}",
                    f"w_{color}",
                    f"kappa_{color}",
                ]
            )

        missing = [name for name in required if name not in index]
        if missing:
            old_rgb_columns = all(
                name in index
                for name in (
                    "valid_R", "h_R", "w_R", "kappa_R",
                    "valid_G", "h_G", "w_G", "kappa_G",
                    "valid_B", "h_B", "w_B", "kappa_B",
                )
            )
            if old_rgb_columns and any("Gr" in x or "Gb" in x for x in missing):
                raise ValueError(
                    f"{path}: 这是旧的 R/G/B 三通道 field_data CSV，"
                    "无法恢复 Gr/Gb。请用四通道统计程序重新导出 "
                    "R/Gr/Gb/B 的 full-field CSV。"
                )
            raise KeyError(f"{path}: 缺少必要列：{missing}")
        return index

    @staticmethod
    def _infer_spatial_shape(
        path: Path,
        row_index: int,
        col_index: int,
    ) -> Tuple[int, int]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            coords = np.loadtxt(
                path,
                delimiter=",",
                skiprows=1,
                usecols=(row_index, col_index),
                dtype=np.int64,
                ndmin=2,
            )
        if coords.size == 0:
            raise ValueError(f"{path}: CSV 没有数据行")
        if np.any(coords < 0):
            raise ValueError(f"{path}: qpd_row/qpd_col 不允许负值")
        return int(np.max(coords[:, 0])) + 1, int(np.max(coords[:, 1])) + 1

    @staticmethod
    def _cache_path(path: Path) -> Path:
        return path.with_suffix(path.suffix + ".npz")

    def _load_entry_cache(
        self,
        path: Path,
        distance: str,
        aperture: str,
    ) -> Optional[HwkEntry]:
        if not self.use_cache:
            return None
        cache_path = self._cache_path(path)
        try:
            if not cache_path.is_file() or cache_path.stat().st_mtime_ns < path.stat().st_mtime_ns:
                return None
            with np.load(cache_path, allow_pickle=False) as cache:
                version = int(np.asarray(cache["cache_version"]).reshape(-1)[0])
                if version != self.CACHE_VERSION:
                    return None
                h = np.asarray(cache["h"], dtype=np.float32)
                w = np.asarray(cache["w"], dtype=np.float32)
                kappa = np.asarray(cache["kappa"], dtype=np.float32)
                valid = np.asarray(cache["valid"], dtype=bool)
        except (OSError, ValueError, KeyError, EOFError):
            return None

        if h.ndim != 3 or h.shape[-1] != len(CFA_COLORS):
            return None
        if w.shape != h.shape or kappa.shape != h.shape or valid.shape != h.shape:
            return None
        return HwkEntry(
            distance=distance,
            aperture=aperture,
            source_path=path,
            h=h,
            w=w,
            kappa=kappa,
            valid=valid,
        )

    def _write_entry_cache(self, entry: HwkEntry) -> None:
        if not self.use_cache:
            return
        cache_path = self._cache_path(entry.source_path)
        temp_path = Path(str(cache_path) + ".tmp.npz")
        try:
            np.savez(
                temp_path,
                cache_version=np.asarray([self.CACHE_VERSION], dtype=np.int32),
                h=entry.h,
                w=entry.w,
                kappa=entry.kappa,
                valid=entry.valid,
            )
            temp_path.replace(cache_path)
        except OSError as exc:
            warnings.warn(
                f"无法写入 HWK 缓存 {cache_path}: {exc}",
                RuntimeWarning,
            )
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass

    def _load_entry(
        self,
        path: Path,
        manifest_row: Optional[Mapping[str, str]],
    ) -> HwkEntry:
        distance, aperture, manifest_height, manifest_width, csv_mode = self._read_metadata(
            path, manifest_row
        )
        cached = self._load_entry_cache(path, distance, aperture)
        if cached is not None:
            if manifest_height is not None and cached.spatial_shape[0] != manifest_height:
                cached = None
            if manifest_width is not None and cached is not None and cached.spatial_shape[1] != manifest_width:
                cached = None
        if cached is not None:
            return cached

        header = self._read_header(path)
        index = self._validate_header(path, header)

        height, width = manifest_height, manifest_width
        if height is None or width is None:
            height, width = self._infer_spatial_shape(
                path, index["qpd_row"], index["qpd_col"]
            )
            if csv_mode not in ("unknown", "all_positions"):
                warnings.warn(
                    f"{path}: 未从 manifest 获得完整 qpd_height/qpd_width，"
                    "且 CSV 可能只保存有效点；由最大坐标推断的尺寸可能偏小。",
                    RuntimeWarning,
                )

        h = np.full((height, width, len(CFA_COLORS)), np.nan, dtype=np.float32)
        w = np.full_like(h, np.nan)
        kappa = np.full_like(h, np.nan)
        valid = np.zeros((height, width, len(CFA_COLORS)), dtype=bool)
        position_seen = np.zeros((height, width), dtype=bool)

        selected_columns = ["qpd_row", "qpd_col"]
        for color in CFA_COLORS:
            selected_columns.extend(
                [
                    f"valid_{color}",
                    f"h_{color}",
                    f"w_{color}",
                    f"kappa_{color}",
                ]
            )
        usecols = tuple(index[name] for name in selected_columns)

        loaded_rows = 0
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            file.readline()  # 跳过表头
            while True:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    chunk = np.loadtxt(
                        file,
                        delimiter=",",
                        usecols=usecols,
                        dtype=np.float64,
                        ndmin=2,
                        max_rows=self.csv_chunk_rows,
                    )
                if chunk.size == 0:
                    break

                row_f = chunk[:, 0]
                col_f = chunk[:, 1]
                if not np.all(np.isfinite(row_f)) or not np.all(np.isfinite(col_f)):
                    raise ValueError(f"{path}: qpd_row/qpd_col 包含 NaN 或 Inf")
                if not np.all(row_f == np.floor(row_f)) or not np.all(col_f == np.floor(col_f)):
                    raise ValueError(f"{path}: qpd_row/qpd_col 必须为整数")

                rows = row_f.astype(np.int64)
                cols = col_f.astype(np.int64)
                if (
                    np.any(rows < 0)
                    or np.any(rows >= height)
                    or np.any(cols < 0)
                    or np.any(cols >= width)
                ):
                    raise ValueError(
                        f"{path}: QPD 坐标超出参数场尺寸 {(height, width)}"
                    )

                flat = rows * width + cols
                if np.unique(flat).size != flat.size:
                    raise ValueError(f"{path}: 同一 CSV chunk 内存在重复 QPD 坐标")
                if np.any(position_seen[rows, cols]):
                    raise ValueError(f"{path}: CSV 中存在跨 chunk 的重复 QPD 坐标")
                position_seen[rows, cols] = True

                offset = 2
                for channel_index, color in enumerate(CFA_COLORS):
                    valid_flag = chunk[:, offset] > 0.5
                    h_value = chunk[:, offset + 1]
                    w_value = chunk[:, offset + 2]
                    k_value = chunk[:, offset + 3]
                    finite = (
                        np.isfinite(h_value)
                        & np.isfinite(w_value)
                        & np.isfinite(k_value)
                    )
                    bad_flagged_rows = valid_flag & ~finite
                    if np.any(bad_flagged_rows):
                        count = int(np.count_nonzero(bad_flagged_rows))
                        raise ValueError(
                            f"{path}: {color} 有 {count} 行 valid=1 但 h/w/kappa 非有限"
                        )

                    channel_valid = valid_flag & finite
                    if np.any(channel_valid):
                        vr = rows[channel_valid]
                        vc = cols[channel_valid]
                        h[vr, vc, channel_index] = h_value[channel_valid].astype(np.float32)
                        w[vr, vc, channel_index] = w_value[channel_valid].astype(np.float32)
                        kappa[vr, vc, channel_index] = k_value[channel_valid].astype(np.float32)
                        valid[vr, vc, channel_index] = True
                    offset += 4

                loaded_rows += int(chunk.shape[0])

        if loaded_rows == 0:
            raise ValueError(f"{path}: CSV 没有数据行")

        if csv_mode == "all_positions" and not np.all(position_seen):
            missing_count = int(position_seen.size - np.count_nonzero(position_seen))
            raise ValueError(
                f"{path}: manifest 声明 csv_mode=all_positions，"
                f"但缺少 {missing_count} 个 QPD 坐标"
            )

        entry = HwkEntry(
            distance=distance,
            aperture=aperture,
            source_path=path,
            h=h,
            w=w,
            kappa=kappa,
            valid=valid,
        )
        self._write_entry_cache(entry)
        return entry

    @property
    def available_conditions(self) -> Tuple[ConditionKey, ...]:
        return tuple(sorted(self._entries.keys()))

    @property
    def distances(self) -> Tuple[str, ...]:
        return tuple(sorted({key[0] for key in self._entries}))

    @property
    def apertures(self) -> Tuple[str, ...]:
        return tuple(sorted({key[1] for key in self._entries}))

    def get(self, distance: str, aperture: str) -> HwkEntry:
        key = (str(distance), str(aperture))
        if key not in self._entries:
            available = ", ".join(f"{d}/{a}" for d, a in self.available_conditions)
            raise KeyError(f"不存在条件 {key}；可用条件：{available}")
        return self._entries[key]

    def select(
        self,
        rng: np.random.Generator,
        *,
        distance: Optional[str] = None,
        aperture: Optional[str] = None,
        distance_candidates: Optional[Sequence[str]] = None,
        aperture_candidates: Optional[Sequence[str]] = None,
        condition_weights: Optional[Mapping[Union[str, ConditionKey], float]] = None,
    ) -> HwkEntry:
        allowed_distances = (
            {str(distance)}
            if distance is not None
            else ({str(x) for x in distance_candidates} if distance_candidates else None)
        )
        allowed_apertures = (
            {str(aperture)}
            if aperture is not None
            else ({str(x) for x in aperture_candidates} if aperture_candidates else None)
        )

        candidates: List[HwkEntry] = []
        for (d, a), entry in self._entries.items():
            if allowed_distances is not None and d not in allowed_distances:
                continue
            if allowed_apertures is not None and a not in allowed_apertures:
                continue
            candidates.append(entry)

        if not candidates:
            raise ValueError(
                "没有满足选择条件的标定数据："
                f"distance={distance}, aperture={aperture}, "
                f"distance_candidates={distance_candidates}, "
                f"aperture_candidates={aperture_candidates}"
            )

        if len(candidates) == 1:
            return candidates[0]
        if condition_weights is None:
            return candidates[int(rng.integers(0, len(candidates)))]

        weights = []
        for entry in candidates:
            tuple_key = entry.key
            text_key = f"{entry.distance}|{entry.aperture}"
            weights.append(
                float(condition_weights.get(tuple_key, condition_weights.get(text_key, 0.0)))
            )

        probability = np.asarray(weights, dtype=np.float64)
        if np.any(probability < 0):
            raise ValueError("condition_weights 不允许负值")
        if probability.sum() <= 0:
            raise ValueError("候选条件的 condition_weights 总和必须大于 0")
        probability /= probability.sum()
        return candidates[int(rng.choice(len(candidates), p=probability))]

    @staticmethod
    def prepare_full_field(
        entry: HwkEntry,
        *,
        invalid_policy: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        h = entry.h.astype(np.float32, copy=True)
        w = entry.w.astype(np.float32, copy=True)
        kappa = entry.kappa.astype(np.float32, copy=True)
        valid = entry.valid.copy()

        policy = str(invalid_policy).lower()
        if policy == "raise":
            if not np.all(valid):
                invalid_count = int(valid.size - np.count_nonzero(valid))
                raise ValueError(f"完整参数场包含 {invalid_count} 个无效通道位置")
        elif policy == "ideal":
            h[~valid] = 1.0
            w[~valid] = 1.0
            kappa[~valid] = 0.0
        else:
            raise ValueError("invalid_policy 当前支持 'ideal' 或 'raise'")

        return h, w, kappa, valid

    @staticmethod
    def prepare_field(
        entry: HwkEntry,
        target_shape: Tuple[int, int],
        *,
        invalid_policy: str,
        field_policy: str = "center_crop",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """提取与目标 QPD 网格一致的 HWK 参数场。"""
        target_height, target_width = map(int, target_shape)
        source_height, source_width = entry.spatial_shape
        if target_height <= 0 or target_width <= 0:
            raise ValueError("目标 HWK 参数场尺寸必须为正数")
        if target_height > source_height or target_width > source_width:
            raise ValueError(
                "目标 QPD 网格大于完整 HWK 标定场："
                f"target={(target_height, target_width)}, "
                f"source={(source_height, source_width)}"
            )

        policy = str(field_policy).lower()
        if policy != "center_crop":
            raise ValueError("field_policy 当前仅支持 center_crop")
        y0 = (source_height - target_height) // 2
        x0 = (source_width - target_width) // 2
        y1 = y0 + target_height
        x1 = x0 + target_width

        h = entry.h[y0:y1, x0:x1].astype(np.float32, copy=True)
        w = entry.w[y0:y1, x0:x1].astype(np.float32, copy=True)
        kappa = entry.kappa[y0:y1, x0:x1].astype(np.float32, copy=True)
        valid = entry.valid[y0:y1, x0:x1].copy()

        invalid = ~valid
        invalid_mode = str(invalid_policy).lower()
        if invalid_mode == "raise":
            if np.any(invalid):
                raise ValueError(
                    f"裁剪后的 HWK 参数场包含 {int(np.count_nonzero(invalid))} 个无效通道位置"
                )
        elif invalid_mode == "ideal":
            h[invalid] = 1.0
            w[invalid] = 1.0
            kappa[invalid] = 0.0
        else:
            raise ValueError("invalid_policy 当前支持 'ideal' 或 'raise'")

        crop_meta = {
            "policy": policy,
            "source_shape": (source_height, source_width),
            "target_shape": (target_height, target_width),
            "origin_qpd": (y0, x0),
            "end_qpd_exclusive": (y1, x1),
            "origin_raw": (y0 * 4, x0 * 4),
            "raw_shape": (target_height * 4, target_width * 4),
        }
        return h, w, kappa, valid, crop_meta


# ============================================================
# 3. QPD 仿真器
# ============================================================
DEFAULT_CONFIG: Dict[str, Any] = {
    "distance": None,
    "aperture": None,
    "distance_candidates": None,
    "aperture_candidates": None,
    "condition_weights": None,

    "invalid_policy": "ideal",
    "field_policy": "center_crop",

    # h/w/kappa 微小随机扰动。
    "prob_hwk_jitter": 1.0,
    # sample: 全样本共享；channel: R/Gr/Gb/B 各一个；qpd: 逐位置逐通道。
    "hwk_jitter_scope": "channel",
    "hwk_jitter_distribution": "normal",  # normal / uniform
    "hwk_jitter_std": {
        "h": (0.0, 0.005),
        "w": (0.0, 0.005),
        "kappa": (0.0, 0.002),
    },
    "hwk_jitter_clip_sigma": 3.0,
    "jitter_invalid_points": False,
    "hw_bounds": (0.0, 2.0),

    # 可选随机 residual mix。
    "prob_rdm_mix": 0.3,
    "rdm_mix_strength": (0.0, 0.05),
    # rdm_mix_strength 表示视场角落处的总泄漏强度；中心为其一定比例。
    "rdm_mix_center_ratio": (0.2, 0.5),
    "rdm_mix_field_power": (1.0, 2.0),
    "rdm_mix_chunk_rows": 64,
    "prob_flatten": 0.8,
    "rdm_mix_channel_mode": "independent",  # independent / shared

    "require_nonnegative_input": True,
    "input_negative_tolerance": 1e-7,
    "output_clip": None,
}


class QpdHwkSimulator:
    """以四通道逐位置 h/w/kappa 为主分支的完整视场 QPD 仿真器。"""

    def __init__(
        self,
        hwk_bank: Union[HwkBank, str, Path, Sequence[Union[str, Path]]],
        *,
        config: Optional[Mapping[str, Any]] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.hwk_bank = hwk_bank if isinstance(hwk_bank, HwkBank) else HwkBank(hwk_bank)
        self.config = self._merge_config(config)
        self.rng = np.random.default_rng(seed)
        self.qpd_cell = 2
        self.cfa_channels = 4
        self.last_meta: Optional[Dict[str, Any]] = None

    @staticmethod
    def _merge_config(config: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        merged = dict(DEFAULT_CONFIG)
        merged["hwk_jitter_std"] = dict(DEFAULT_CONFIG["hwk_jitter_std"])
        if config is not None:
            for key, value in config.items():
                if key == "hwk_jitter_std":
                    merged["hwk_jitter_std"].update(dict(value))
                else:
                    merged[key] = value
        return merged

    @staticmethod
    def _sample_scalar_or_range(value: ScalarOrRange, rng: np.random.Generator) -> float:
        if np.isscalar(value):
            return float(value)
        sequence = tuple(float(x) for x in value)
        if len(sequence) != 2:
            raise ValueError(f"参数范围必须为长度 2，实际为 {sequence}")
        low, high = sequence
        if high < low:
            raise ValueError(f"参数范围顺序错误：{sequence}")
        return float(rng.uniform(low, high))

    @staticmethod
    def _legal_kappa_bounds(h: np.ndarray, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        lower = np.maximum(-h * w, -(2.0 - h) * (2.0 - w))
        upper = np.minimum((2.0 - h) * w, h * (2.0 - w))
        return lower, upper

    def _noise_shape(self, spatial_shape: Tuple[int, int], scope: str) -> Tuple[int, int, int]:
        scope = scope.lower()
        if scope == "sample":
            return 1, 1, 1
        if scope == "channel":
            return 1, 1, self.cfa_channels
        if scope == "qpd":
            return spatial_shape[0], spatial_shape[1], self.cfa_channels
        raise ValueError("hwk_jitter_scope 必须为 sample/channel/qpd")

    def _sample_jitter(self, scale: float, shape: Tuple[int, int, int]) -> np.ndarray:
        if scale <= 0.0:
            return np.zeros(shape, dtype=np.float32)

        distribution = str(self.config["hwk_jitter_distribution"]).lower()
        if distribution == "normal":
            noise = self.rng.normal(0.0, scale, size=shape)
            clip_sigma = self.config.get("hwk_jitter_clip_sigma")
            if clip_sigma is not None:
                limit = abs(float(clip_sigma)) * scale
                noise = np.clip(noise, -limit, limit)
        elif distribution == "uniform":
            noise = self.rng.uniform(-scale, scale, size=shape)
        else:
            raise ValueError("hwk_jitter_distribution 必须为 normal 或 uniform")
        return noise.astype(np.float32)

    def perturb_hwk(
        self,
        h: np.ndarray,
        w: np.ndarray,
        kappa: np.ndarray,
        valid: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        h_out = h.astype(np.float32, copy=True)
        w_out = w.astype(np.float32, copy=True)
        k_out = kappa.astype(np.float32, copy=True)

        applied = self.rng.random() < float(self.config["prob_hwk_jitter"])
        meta: Dict[str, Any] = {
            "applied": bool(applied),
            "scope": str(self.config["hwk_jitter_scope"]),
            "sigma_h": 0.0,
            "sigma_w": 0.0,
            "sigma_kappa": 0.0,
            "kappa_clipped_fraction": 0.0,
        }
        if not applied:
            return h_out, w_out, k_out, meta

        std_cfg = self.config["hwk_jitter_std"]
        sigma_h = self._sample_scalar_or_range(std_cfg["h"], self.rng)
        sigma_w = self._sample_scalar_or_range(std_cfg["w"], self.rng)
        sigma_k = self._sample_scalar_or_range(std_cfg["kappa"], self.rng)
        shape = self._noise_shape(h.shape[:2], str(self.config["hwk_jitter_scope"]))

        h_out += self._sample_jitter(sigma_h, shape)
        w_out += self._sample_jitter(sigma_w, shape)
        k_out += self._sample_jitter(sigma_k, shape)

        # 无效位置回退到理想响应后，默认不再施加随机扰动。
        if not bool(self.config.get("jitter_invalid_points", False)):
            h_out[~valid] = 1.0
            w_out[~valid] = 1.0
            k_out[~valid] = 0.0

        hw_low, hw_high = map(float, self.config["hw_bounds"])
        if hw_high < hw_low:
            raise ValueError("hw_bounds 顺序错误")
        h_out = np.clip(h_out, hw_low, hw_high)
        w_out = np.clip(w_out, hw_low, hw_high)

        k_low, k_high = self._legal_kappa_bounds(h_out, w_out)
        k_before = k_out.copy()
        k_out = np.clip(k_out, k_low, k_high)

        meta.update(
            {
                "sigma_h": sigma_h,
                "sigma_w": sigma_w,
                "sigma_kappa": sigma_k,
                "kappa_clipped_fraction": float(np.mean(np.abs(k_out - k_before) > 1e-12)),
            }
        )
        return h_out, w_out, k_out, meta

    @staticmethod
    def _split_qpd(cfa_planes: np.ndarray) -> Tuple[np.ndarray, ...]:
        return (
            cfa_planes[0::2, 0::2, :],
            cfa_planes[0::2, 1::2, :],
            cfa_planes[1::2, 0::2, :],
            cfa_planes[1::2, 1::2, :],
        )

    @staticmethod
    def _merge_qpd(tl: np.ndarray, tr: np.ndarray, bl: np.ndarray, br: np.ndarray) -> np.ndarray:
        hq, wq, channels = tl.shape
        output = np.empty((hq * 2, wq * 2, channels), dtype=tl.dtype)
        output[0::2, 0::2, :] = tl
        output[0::2, 1::2, :] = tr
        output[1::2, 0::2, :] = bl
        output[1::2, 1::2, :] = br
        return output

    @staticmethod
    def apply_hw_reorder(cfa_planes: np.ndarray, h: np.ndarray, w: np.ndarray) -> np.ndarray:
        src_tl, src_tr, src_bl, src_br = QpdHwkSimulator._split_qpd(cfa_planes)

        h_l = np.minimum(h, 1.0)
        h_r = np.maximum(h - 1.0, 0.0)
        one_minus_h = np.maximum(1.0 - h, 0.0)
        two_minus_h = np.minimum(2.0 - h, 1.0)

        w_t = np.minimum(w, 1.0)
        w_b = np.maximum(w - 1.0, 0.0)
        one_minus_w = np.maximum(1.0 - w, 0.0)
        two_minus_w = np.minimum(2.0 - w, 1.0)

        out_tl = (
            src_tl * (h_l * w_t)
            + src_tr * (h_r * w_t)
            + src_bl * (h_l * w_b)
            + src_br * (h_r * w_b)
        )
        out_tr = (
            src_tl * (one_minus_h * w_t)
            + src_tr * (two_minus_h * w_t)
            + src_bl * (one_minus_h * w_b)
            + src_br * (two_minus_h * w_b)
        )
        out_bl = (
            src_tl * (h_l * one_minus_w)
            + src_tr * (h_r * one_minus_w)
            + src_bl * (h_l * two_minus_w)
            + src_br * (h_r * two_minus_w)
        )
        out_br = (
            src_tl * (one_minus_h * one_minus_w)
            + src_tr * (two_minus_h * one_minus_w)
            + src_bl * (one_minus_h * two_minus_w)
            + src_br * (two_minus_h * two_minus_w)
        )
        return QpdHwkSimulator._merge_qpd(out_tl, out_tr, out_bl, out_br)

    @staticmethod
    def apply_kappa_transfer(
        src_before_reorder: np.ndarray,
        hw_output: np.ndarray,
        kappa: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        src_tl, src_tr, src_bl, src_br = QpdHwkSimulator._split_qpd(src_before_reorder)
        out_tl, out_tr, out_bl, out_br = [
            item.copy() for item in QpdHwkSimulator._split_qpd(hw_output)
        ]

        block_mean = 0.25 * (src_tl + src_tr + src_bl + src_br)

        # kappa > 0: TR -> TL，BL -> BR。
        target_pos = np.maximum(kappa, 0.0) * block_mean
        transfer_pos = np.minimum(target_pos, np.minimum(out_tr, out_bl))
        out_tl += transfer_pos
        out_br += transfer_pos
        out_tr -= transfer_pos
        out_bl -= transfer_pos

        # kappa < 0: TL -> TR，BR -> BL。
        target_neg = np.maximum(-kappa, 0.0) * block_mean
        transfer_neg = np.minimum(target_neg, np.minimum(out_tl, out_br))
        out_tl -= transfer_neg
        out_br -= transfer_neg
        out_tr += transfer_neg
        out_bl += transfer_neg

        target = target_pos + target_neg
        actual = transfer_pos + transfer_neg
        limited = target > actual + 1e-12

        result = QpdHwkSimulator._merge_qpd(out_tl, out_tr, out_bl, out_br)
        return result, {
            "limited_fraction": float(np.mean(limited)),
            "mean_target_transfer": float(np.mean(target)),
            "mean_actual_transfer": float(np.mean(actual)),
        }

    def apply_hwk(
        self,
        cfa_planes: np.ndarray,
        h: np.ndarray,
        w: np.ndarray,
        kappa: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        hw_output = self.apply_hw_reorder(cfa_planes, h, w)
        output, kappa_meta = self.apply_kappa_transfer(cfa_planes, hw_output, kappa)
        return output, {"kappa_transfer": kappa_meta}

    def _generate_rdm_mix_chunk(
        self,
        strength: np.ndarray,
        *,
        flatten: bool,
        channel_mode: str,
    ) -> np.ndarray:
        """生成一个 QPD 行块的空间变化 residual mix 矩阵。"""
        local_strength = np.asarray(strength, dtype=np.float32)
        if local_strength.ndim != 2:
            raise ValueError("strength 应为 (Hq,Wq)")
        if np.any(local_strength < 0.0) or np.any(local_strength > 1.0):
            raise ValueError("rdm mix 局部强度必须位于 [0,1]")

        mode = str(channel_mode).lower()
        if mode not in {"independent", "shared"}:
            raise ValueError("rdm_mix_channel_mode 必须为 independent 或 shared")

        hq, wq = local_strength.shape
        generated_channels = 1 if mode == "shared" else self.cfa_channels
        matrix_generated = np.zeros(
            (hq, wq, len(PD_ORDER), len(PD_ORDER), generated_channels),
            dtype=np.float32,
        )
        strength_4d = local_strength[:, :, None]

        if flatten:
            # 二维 QPD 几何：H/V/D 分别表示水平、垂直和对角串扰。
            proportions = self.rng.dirichlet(
                np.ones(3), size=(hq, wq, generated_channels)
            ).astype(np.float32)
            horizontal = strength_4d * proportions[..., 0]
            vertical = strength_4d * proportions[..., 1]
            diagonal = strength_4d * proportions[..., 2]
            stay = 1.0 - strength_4d

            matrix_generated[:, :, 0, 0, :] = stay
            matrix_generated[:, :, 0, 1, :] = horizontal
            matrix_generated[:, :, 0, 2, :] = vertical
            matrix_generated[:, :, 0, 3, :] = diagonal

            matrix_generated[:, :, 1, 0, :] = horizontal
            matrix_generated[:, :, 1, 1, :] = stay
            matrix_generated[:, :, 1, 2, :] = diagonal
            matrix_generated[:, :, 1, 3, :] = vertical

            matrix_generated[:, :, 2, 0, :] = vertical
            matrix_generated[:, :, 2, 1, :] = diagonal
            matrix_generated[:, :, 2, 2, :] = stay
            matrix_generated[:, :, 2, 3, :] = horizontal

            matrix_generated[:, :, 3, 0, :] = diagonal
            matrix_generated[:, :, 3, 1, :] = vertical
            matrix_generated[:, :, 3, 2, :] = horizontal
            matrix_generated[:, :, 3, 3, :] = stay
        else:
            # 每个输入 PD 独立决定泄漏到另外三个输出 PD 的比例。
            for input_pd in range(len(PD_ORDER)):
                proportions = self.rng.dirichlet(
                    np.ones(3), size=(hq, wq, generated_channels)
                ).astype(np.float32)
                matrix_generated[:, :, input_pd, input_pd, :] = 1.0 - strength_4d
                offdiag = [index for index in range(len(PD_ORDER)) if index != input_pd]
                for proportion_index, output_pd in enumerate(offdiag):
                    matrix_generated[:, :, input_pd, output_pd, :] = (
                        strength_4d * proportions[..., proportion_index]
                    )

        if mode == "shared":
            return np.broadcast_to(
                matrix_generated,
                (hq, wq, len(PD_ORDER), len(PD_ORDER), self.cfa_channels),
            )
        return matrix_generated

    @staticmethod
    def _radial_rdm_strength_field(
        spatial_shape: Tuple[int, int],
        *,
        edge_strength: float,
        center_ratio: float,
        field_power: float,
    ) -> np.ndarray:
        """构造从光轴中心向视场角落连续增强的混合强度场。"""
        hq, wq = map(int, spatial_shape)
        if hq <= 0 or wq <= 0:
            raise ValueError("rdm mix 空间尺寸必须为正数")
        if not (0.0 <= edge_strength <= 1.0):
            raise ValueError("rdm_mix_strength 必须位于 [0,1]")
        if not (0.0 <= center_ratio <= 1.0):
            raise ValueError("rdm_mix_center_ratio 必须位于 [0,1]")
        if field_power <= 0.0:
            raise ValueError("rdm_mix_field_power 必须大于 0")

        yy = np.linspace(-1.0, 1.0, hq, dtype=np.float32)[:, None]
        xx = np.linspace(-1.0, 1.0, wq, dtype=np.float32)[None, :]
        radius = np.minimum(np.sqrt(xx * xx + yy * yy) / np.sqrt(2.0), 1.0)
        profile = center_ratio + (1.0 - center_ratio) * np.power(radius, field_power)
        return (edge_strength * profile).astype(np.float32)

    def generate_rdm_mix_matrix(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """生成单个 QPD 使用的 (输入PD,输出PD,CFA) residual mix。"""
        strength = self._sample_scalar_or_range(self.config["rdm_mix_strength"], self.rng)
        if not (0.0 <= strength <= 1.0):
            raise ValueError("rdm_mix_strength 必须位于 [0,1]")
        flatten = self.rng.random() < float(self.config["prob_flatten"])
        mode = str(self.config["rdm_mix_channel_mode"]).lower()
        matrix = self._generate_rdm_mix_chunk(
            np.asarray([[strength]], dtype=np.float32),
            flatten=bool(flatten),
            channel_mode=mode,
        )[0, 0].copy()
        row_sum = np.sum(matrix, axis=1)
        return matrix, {
            "matrix_shape": tuple(matrix.shape),
            "cfa_order": CFA_COLORS,
            "strength": strength,
            "flatten": bool(flatten),
            "matrix_mode": "flatten" if flatten else "random",
            "channel_mode": mode,
            "row_sum_error": float(np.max(np.abs(row_sum - 1.0))),
            "minimum_coefficient": float(np.min(matrix)),
            "maximum_coefficient": float(np.max(matrix)),
        }

    def apply_spatial_rdm_mix(
        self,
        cfa_planes: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """在 HWK 输出上逐 QPD 应用中心弱、边缘强的独立 residual mix。"""
        src = np.asarray(cfa_planes, dtype=np.float32)
        if src.ndim != 3 or src.shape[-1] != self.cfa_channels:
            raise ValueError(f"cfa_planes 应为 (Hc,Wc,4)，实际为 {src.shape}")
        if src.shape[0] % 2 != 0 or src.shape[1] % 2 != 0:
            raise ValueError("cfa_planes 的高度和宽度必须为偶数")

        edge_strength = self._sample_scalar_or_range(
            self.config["rdm_mix_strength"], self.rng
        )
        center_ratio = self._sample_scalar_or_range(
            self.config["rdm_mix_center_ratio"], self.rng
        )
        field_power = self._sample_scalar_or_range(
            self.config["rdm_mix_field_power"], self.rng
        )
        flatten = self.rng.random() < float(self.config["prob_flatten"])
        channel_mode = str(self.config["rdm_mix_channel_mode"]).lower()
        chunk_rows = int(self.config["rdm_mix_chunk_rows"])
        if chunk_rows <= 0:
            raise ValueError("rdm_mix_chunk_rows 必须为正数")

        blocks = np.stack(self._split_qpd(src), axis=-1)  # Hq,Wq,CFA,输入PD
        hq, wq = blocks.shape[:2]
        strength_field = self._radial_rdm_strength_field(
            (hq, wq),
            edge_strength=edge_strength,
            center_ratio=center_ratio,
            field_power=field_power,
        )
        output_blocks = np.empty_like(blocks)
        max_row_sum_error = 0.0
        minimum_coefficient = 1.0
        maximum_coefficient = 0.0

        for row_start in range(0, hq, chunk_rows):
            row_end = min(row_start + chunk_rows, hq)
            matrix = self._generate_rdm_mix_chunk(
                strength_field[row_start:row_end],
                flatten=bool(flatten),
                channel_mode=channel_mode,
            )
            row_sum = np.sum(matrix, axis=3)
            max_row_sum_error = max(
                max_row_sum_error,
                float(np.max(np.abs(row_sum - 1.0))),
            )
            minimum_coefficient = min(minimum_coefficient, float(np.min(matrix)))
            maximum_coefficient = max(maximum_coefficient, float(np.max(matrix)))
            output_blocks[row_start:row_end] = np.einsum(
                "hwci,hwijc->hwcj",
                blocks[row_start:row_end],
                matrix,
                optimize=True,
            )

        output = self._merge_qpd(
            output_blocks[..., 0],
            output_blocks[..., 1],
            output_blocks[..., 2],
            output_blocks[..., 3],
        )
        return output, {
            "matrix_shape_per_qpd": (4, 4, self.cfa_channels),
            "matrix_mode": "flatten" if flatten else "random",
            "flatten": bool(flatten),
            "channel_mode": channel_mode,
            "per_qpd_independent": True,
            "field_profile": "radial_center_weak_edge_strong",
            "edge_strength": edge_strength,
            "center_ratio": center_ratio,
            "center_strength": edge_strength * center_ratio,
            "field_power": field_power,
            "strength_min": float(np.min(strength_field)),
            "strength_mean": float(np.mean(strength_field)),
            "strength_max": float(np.max(strength_field)),
            "row_sum_error": max_row_sum_error,
            "minimum_coefficient": minimum_coefficient,
            "maximum_coefficient": maximum_coefficient,
            "chunk_rows": chunk_rows,
            "full_matrix_materialized": False,
        }

    @staticmethod
    def validate_rdm_mix_matrix(
        mix_matrix: np.ndarray,
        *,
        atol: float = 1e-6,
    ) -> np.ndarray:
        """检查并返回 float32 的 (4,4,4) 行随机 residual mix 矩阵。"""
        matrix = np.asarray(mix_matrix, dtype=np.float32)
        expected_shape = (len(PD_ORDER), len(PD_ORDER), len(CFA_COLORS))
        if matrix.shape != expected_shape:
            raise ValueError(
                "rdm_mix_matrix 形状错误："
                f"期望 {expected_shape} = 输入PD×输出PD×CFA通道，"
                f"实际为 {matrix.shape}"
            )
        if not np.all(np.isfinite(matrix)):
            raise ValueError("rdm_mix_matrix 包含 NaN 或 Inf")
        if float(np.min(matrix)) < -float(atol):
            raise ValueError(
                "rdm_mix_matrix 必须非负；"
                f"当前最小值为 {float(np.min(matrix))}"
            )

        # 极小负数视作浮点误差修正为 0，再严格检查行和。
        matrix = np.maximum(matrix, 0.0)
        row_sum = np.sum(matrix, axis=1)
        max_error = float(np.max(np.abs(row_sum - 1.0)))
        if max_error > float(atol):
            raise ValueError(
                "rdm_mix_matrix 每个输入PD对应的行和必须为1；"
                f"当前最大误差为 {max_error}"
            )
        return matrix

    @staticmethod
    def apply_rdm_mix(cfa_planes: np.ndarray, mix_matrix: np.ndarray) -> np.ndarray:
        """
        对四 CFA 平面中的每个 2x2 QPD 单元应用 residual mix。

        cfa_planes : (Hc, Wc, 4)，最后一维为 R/Gr/Gb/B
        mix_matrix : (4, 4, 4)，输入PD×输出PD×CFA通道
        """
        src = np.asarray(cfa_planes)
        if src.ndim != 3 or src.shape[-1] != len(CFA_COLORS):
            raise ValueError(f"cfa_planes 应为 (Hc,Wc,4)，实际为 {src.shape}")
        if src.shape[0] % 2 != 0 or src.shape[1] % 2 != 0:
            raise ValueError("cfa_planes 的高度和宽度必须为偶数")

        matrix = QpdHwkSimulator.validate_rdm_mix_matrix(mix_matrix)
        tl, tr, bl, br = QpdHwkSimulator._split_qpd(src)
        blocks = np.stack((tl, tr, bl, br), axis=-1)  # Hq,Wq,CFA,输入PD
        output_blocks = np.einsum(
            "hwci,ijc->hwcj",
            blocks,
            matrix,
            optimize=True,
        )
        return QpdHwkSimulator._merge_qpd(
            output_blocks[..., 0],
            output_blocks[..., 1],
            output_blocks[..., 2],
            output_blocks[..., 3],
        )

    @staticmethod
    def _energy_error(before: np.ndarray, after: np.ndarray) -> float:
        before_blocks = np.stack(QpdHwkSimulator._split_qpd(before), axis=-1)
        after_blocks = np.stack(QpdHwkSimulator._split_qpd(after), axis=-1)
        return float(
            np.max(
                np.abs(
                    np.sum(after_blocks, axis=-1)
                    - np.sum(before_blocks, axis=-1)
                )
            )
        )

    def __call__(
        self,
        cfa_planes: np.ndarray,
        *,
        distance: Optional[str] = None,
        aperture: Optional[str] = None,
        return_meta: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
        src = np.asarray(cfa_planes)
        if src.ndim != 3 or src.shape[2] != self.cfa_channels:
            raise ValueError(f"cfa_planes 应为 (Hc,Wc,4)，实际为 {src.shape}")
        if src.shape[0] % 2 != 0 or src.shape[1] % 2 != 0:
            raise ValueError("四 CFA 平面的高度和宽度必须为偶数")
        if not np.all(np.isfinite(src)):
            raise ValueError("cfa_planes 包含 NaN 或 Inf")

        src_float = src.astype(np.float32, copy=False)
        if self.config["require_nonnegative_input"]:
            tolerance = float(self.config["input_negative_tolerance"])
            minimum = float(np.min(src_float))
            if minimum < -tolerance:
                raise ValueError(
                    "h/w/kappa 分配要求非负线性输入；"
                    f"当前最小值为 {minimum}"
                )
            if minimum < 0.0:
                src_float = np.maximum(src_float, 0.0)

        selected_distance = distance if distance is not None else self.config.get("distance")
        selected_aperture = aperture if aperture is not None else self.config.get("aperture")
        entry = self.hwk_bank.select(
            self.rng,
            distance=selected_distance,
            aperture=selected_aperture,
            distance_candidates=self.config.get("distance_candidates"),
            aperture_candidates=self.config.get("aperture_candidates"),
            condition_weights=self.config.get("condition_weights"),
        )

        expected_qpd_shape = (src.shape[0] // 2, src.shape[1] // 2)
        h, w, kappa, valid, field_crop_meta = self.hwk_bank.prepare_field(
            entry,
            expected_qpd_shape,
            invalid_policy=str(self.config["invalid_policy"]),
            field_policy=str(self.config["field_policy"]),
        )
        h, w, kappa, jitter_meta = self.perturb_hwk(h, w, kappa, valid)
        output, hwk_meta = self.apply_hwk(src_float, h, w, kappa)

        use_mix = self.rng.random() < float(self.config["prob_rdm_mix"])
        mix_meta: Dict[str, Any] = {"applied": bool(use_mix)}
        if use_mix:
            output, generated_meta = self.apply_spatial_rdm_mix(output)
            mix_meta.update(generated_meta)

        energy_error = self._energy_error(src_float, output)

        output_clip = self.config.get("output_clip")
        if output_clip is not None:
            low, high = map(float, output_clip)
            if high < low:
                raise ValueError("output_clip 顺序错误")
            output = np.clip(output, low, high)

        meta: Dict[str, Any] = {
            "input_domain": "linear_camera_cfa_planes",
            "cfa_order": CFA_COLORS,
            "distance": entry.distance,
            "aperture": entry.aperture,
            "source_path": str(entry.source_path),
            "cfa_plane_shape": tuple(src.shape),
            "hwk_source_field_shape": tuple(entry.h.shape),
            "hwk_field_shape": tuple(h.shape),
            "hwk_field_crop": field_crop_meta,
            "valid_fraction_cfa": {
                color: float(np.mean(valid[..., index]))
                for index, color in enumerate(CFA_COLORS)
            },
            "hwk_jitter": jitter_meta,
            "hwk_apply": hwk_meta,
            "rdm_mix": mix_meta,
            "max_qpd_energy_error_before_output_clip": energy_error,
        }

        self.last_meta = meta
        output = output.astype(src_float.dtype, copy=False)
        return (output, meta) if return_meta else output

    def simulate_from_camera_rgb(
        self,
        camera_rgb: np.ndarray,
        *,
        distance: Optional[str] = None,
        aperture: Optional[str] = None,
        return_meta: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
        """
        对线性 camera RGB 执行 Quad RGGB sampling 和 QPD 仿真。

        返回四 CFA 平面，而不是 demosaic RGB。
        """
        cfa_planes = quad_rggb_sample_to_planes(camera_rgb)
        return self(
            cfa_planes,
            distance=distance,
            aperture=aperture,
            return_meta=return_meta,
        )


# ============================================================
# 4. 验证工具
# ============================================================
def reconstruct_flat_response_from_hwk(
    h: np.ndarray,
    w: np.ndarray,
    kappa: np.ndarray,
) -> np.ndarray:
    """返回 (...,4) 的 TL/TR/BL/BR 平场响应。"""
    return np.stack(
        (
            h * w + kappa,
            (2.0 - h) * w - kappa,
            h * (2.0 - w) - kappa,
            (2.0 - h) * (2.0 - w) + kappa,
        ),
        axis=-1,
    )


def validate_simulator_on_uniform_cfa(
    simulator: QpdHwkSimulator,
    *,
    value: float = 0.5,
    distance: Optional[str] = None,
    aperture: Optional[str] = None,
) -> Dict[str, Any]:
    """建议在 prob_hwk_jitter=0、prob_rdm_mix=0 时调用。"""
    entry = simulator.hwk_bank.select(
        simulator.rng,
        distance=distance,
        aperture=aperture,
        distance_candidates=simulator.config.get("distance_candidates"),
        aperture_candidates=simulator.config.get("aperture_candidates"),
        condition_weights=simulator.config.get("condition_weights"),
    )
    hq, wq = entry.spatial_shape
    cfa = np.full((hq * 2, wq * 2, 4), value, dtype=np.float32)
    output, meta = simulator(
        cfa,
        distance=entry.distance,
        aperture=entry.aperture,
        return_meta=True,
    )

    h, w, kappa, _valid = simulator.hwk_bank.prepare_full_field(
        entry,
        invalid_policy=str(simulator.config["invalid_policy"]),
    )
    expected = value * reconstruct_flat_response_from_hwk(h, w, kappa)
    output_blocks = np.stack(QpdHwkSimulator._split_qpd(output), axis=-1)

    return {
        "max_uniform_response_error": float(np.max(np.abs(output_blocks - expected))),
        "max_qpd_energy_error": meta["max_qpd_energy_error_before_output_clip"],
        "condition": (entry.distance, entry.aperture),
    }


if __name__ == "__main__":
    print(
        "本文件假设输入已经是 linear camera RGB。\n"
        "主要接口：quad_rggb_sample_to_planes、quad_rggb_planes_to_mosaic、"
        "HwkBank、QpdHwkSimulator。"
    )
