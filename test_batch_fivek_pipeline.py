import unittest

from batch_fivek_pipeline import is_smaller_than_crop, parse_crop_size


class BatchFiveKPipelineTests(unittest.TestCase):
    def test_parse_crop_size(self):
        self.assertEqual(parse_crop_size("3000x2000"), (3000, 2000))
        self.assertEqual(parse_crop_size("3000 * 2000"), (3000, 2000))

    def test_rejects_invalid_crop_size(self):
        with self.assertRaises(ValueError):
            parse_crop_size("3000")
        with self.assertRaises(ValueError):
            parse_crop_size("0x2000")

    def test_small_image_requires_both_dimensions(self):
        crop = (3000, 2000)
        self.assertFalse(is_smaller_than_crop((3000, 2000), crop))
        self.assertFalse(is_smaller_than_crop((4000, 3000), crop))
        self.assertTrue(is_smaller_than_crop((2999, 3000), crop))
        self.assertTrue(is_smaller_than_crop((4000, 1999), crop))


if __name__ == "__main__":
    unittest.main()
