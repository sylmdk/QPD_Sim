import unittest
from pathlib import Path

import numpy as np

from qpd_hwk_simulator import (
    HwkBank,
    HwkEntry,
    QpdHwkSimulator,
    quad_rggb_planes_to_mosaic,
    quad_rggb_sample_to_planes,
)


def make_bank(shape=(6, 8, 4)):
    ones = np.ones(shape, dtype=np.float32)
    zeros = np.zeros(shape, dtype=np.float32)
    entry = HwkEntry(
        distance="1m",
        aperture="F1.4",
        source_path=Path("synthetic.csv"),
        h=ones,
        w=ones,
        kappa=zeros,
        valid=np.ones(shape, dtype=bool),
    )
    bank = HwkBank.__new__(HwkBank)
    bank._entries = {entry.key: entry}
    return bank, entry


class QpdHwkSimulatorTests(unittest.TestCase):
    def test_quad_rggb_sampling_and_output_size(self):
        camera_rgb = np.zeros((8, 12, 3), dtype=np.float32)
        camera_rgb[..., 0] = 1.0
        camera_rgb[..., 1] = 2.0
        camera_rgb[..., 2] = 3.0

        planes = quad_rggb_sample_to_planes(camera_rgb)
        raw = quad_rggb_planes_to_mosaic(planes)

        expected_tile = np.asarray(
            [
                [1, 1, 2, 2],
                [1, 1, 2, 2],
                [2, 2, 3, 3],
                [2, 2, 3, 3],
            ],
            dtype=np.float32,
        )
        self.assertEqual(planes.shape, (4, 6, 4))
        self.assertEqual(raw.shape, camera_rgb.shape[:2])
        np.testing.assert_array_equal(raw[:4, :4], expected_tile)

        tagged_rgb = np.arange(8 * 12 * 3, dtype=np.float32).reshape(8, 12, 3)
        tagged_raw = quad_rggb_planes_to_mosaic(
            quad_rggb_sample_to_planes(tagged_rgb)
        )
        expected_raw = np.empty((8, 12), dtype=np.float32)
        for block_y, block_x, channel in ((0, 0, 0), (0, 2, 1), (2, 0, 1), (2, 2, 2)):
            expected_raw.reshape(2, 4, 3, 4)[
                :, block_y:block_y + 2, :, block_x:block_x + 2
            ] = tagged_rgb.reshape(2, 4, 3, 4, 3)[
                :, block_y:block_y + 2, :, block_x:block_x + 2, channel
            ]
        np.testing.assert_array_equal(tagged_raw, expected_raw)

    def test_hwk_field_is_center_cropped(self):
        bank, entry = make_bank()
        h, w, kappa, valid, meta = bank.prepare_field(
            entry,
            (4, 6),
            invalid_policy="ideal",
        )

        self.assertEqual(h.shape, (4, 6, 4))
        self.assertEqual(w.shape, h.shape)
        self.assertEqual(kappa.shape, h.shape)
        self.assertEqual(valid.shape, h.shape)
        self.assertEqual(meta["origin_qpd"], (1, 1))
        self.assertEqual(meta["raw_shape"], (16, 24))

    def test_spatial_rdm_is_energy_preserving_and_repeatable(self):
        bank, _entry = make_bank()
        camera_rgb = np.random.default_rng(5).random((16, 24, 3), dtype=np.float32)
        planes = quad_rggb_sample_to_planes(camera_rgb)
        config = {
            "prob_hwk_jitter": 1.0,
            "prob_rdm_mix": 1.0,
            "prob_flatten": 0.0,
            "rdm_mix_chunk_rows": 2,
        }

        first, first_meta = QpdHwkSimulator(bank, config=config, seed=9)(
            planes, return_meta=True
        )
        second, second_meta = QpdHwkSimulator(bank, config=config, seed=9)(
            planes, return_meta=True
        )

        np.testing.assert_array_equal(first, second)
        self.assertLess(first_meta["max_qpd_energy_error_before_output_clip"], 1e-5)
        self.assertEqual(first_meta["rdm_mix"]["matrix_mode"], "random")
        self.assertLess(
            first_meta["rdm_mix"]["center_strength"],
            first_meta["rdm_mix"]["edge_strength"],
        )
        self.assertEqual(first_meta["rdm_mix"], second_meta["rdm_mix"])


if __name__ == "__main__":
    unittest.main()
