

import unittest

from torch.utils.data import TensorDataset, DataLoader
import torch

from torch_fairness.sampler import MLStratifiedBatchSampler


class TestMLStratifiedBatchSampler(unittest.TestCase):
    @classmethod
    def setUp(cls):
        nan = float('nan')
        cls.features = torch.arange(8)[:, None]
        cls.sensitive = torch.tensor([
            [nan, 0, nan],
            [0, 0, nan],
            [0, 1, 0],
            [0, 1, 1],
            [1, nan, 0],
            [nan, 0, 1],
            [1, 1, nan],
            [1, 1, 1]
        ])
        cls.dataset = TensorDataset(cls.features, cls.sensitive)

    @classmethod
    def tearDown(cls):
        cls.features = None
        cls.sensitive = None

    def test_no_overlap(self):
        data_loader = DataLoader(
            dataset=self.dataset,
            batch_sampler=MLStratifiedBatchSampler(self.sensitive, batch_size=4)
        )
        indices = []
        for n_batches, batch in enumerate(data_loader):
            x, _ = batch
            indices += x.squeeze().numpy().tolist()
        n_rows = self.sensitive.shape[0]
        self.assertEqual(n_rows, len(indices))

    def test_batch_size_input(self):
        self.assertRaises(ValueError, MLStratifiedBatchSampler, labels=self.sensitive, batch_size='apple')

    def test_insufficient_batch_size(self):
        self.assertRaises(ValueError, MLStratifiedBatchSampler, labels=self.sensitive, batch_size=1)

    def test_batch_size_warning(self):
        sensitive = torch.tensor([
            [0., 1., 1., 0.],
            [1., 0., 0., 1.],
            [1., 0., float('nan'), float('nan')]
        ])
        self.assertRaises(ValueError, MLStratifiedBatchSampler, labels=sensitive, batch_size=3)

    def test_len_dunder(self):
        batch_sampler = MLStratifiedBatchSampler(self.sensitive, batch_size=4)
        self.assertEqual(self.sensitive.shape[0], len(batch_sampler))

    def test_limited_replacement(self):
        data_loader = DataLoader(
            dataset=self.dataset,
            batch_sampler=MLStratifiedBatchSampler(self.sensitive, batch_size=4, limit_replacement=True)
        )
        indices = []
        for n_batches, batch in enumerate(data_loader):
            x, _ = batch
            indices += x.squeeze().numpy().tolist()
        n_rows = self.sensitive.shape[0]
        self.assertEqual(n_rows, len(indices))
