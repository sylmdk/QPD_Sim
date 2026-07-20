import unittest

from split_qpd_dataset import split_samples


class SplitQpdDatasetTests(unittest.TestCase):
    def setUp(self):
        self.samples = [{"sample_id": f"sample_{index:02d}"} for index in range(20)]

    def test_default_uses_all_samples(self):
        splits = split_samples(self.samples, 0.8, 0.1, seed=2026)
        self.assertEqual(sum(len(items) for items in splits.values()), 20)

    def test_num_samples_limits_reproducible_subset(self):
        first = split_samples(self.samples, 0.8, 0.1, seed=7, num_samples=10)
        second = split_samples(self.samples, 0.8, 0.1, seed=7, num_samples=10)
        self.assertEqual(first, second)
        self.assertEqual([len(first[name]) for name in ("train", "val", "test")], [8, 1, 1])

    def test_num_samples_must_be_in_available_range(self):
        with self.assertRaises(ValueError):
            split_samples(self.samples, 0.8, 0.1, seed=7, num_samples=0)
        with self.assertRaises(ValueError):
            split_samples(self.samples, 0.8, 0.1, seed=7, num_samples=21)


if __name__ == "__main__":
    unittest.main()
