
import unittest

import numpy as np
import pandas as pd

from torch_fairness.resampling import imbalance_ratio
from torch_fairness.resampling import geometric_mean
from torch_fairness.resampling import scumble
from torch_fairness.resampling import adjusted_hamming_distance
from torch_fairness.resampling import MLSMOTE
from torch_fairness.resampling import MLROS
from torch_fairness.resampling import MLRUS
from torch_fairness.resampling import MLeNN


class TestGeometricMean(unittest.TestCase):
    def test_with_nan(self):
        observed = geometric_mean(np.array([2, np.nan]))
        self.assertAlmostEqual(observed, 2., 4)

    def test_unstable(self):
        observed = geometric_mean(np.array([0.0001, 1000]))
        self.assertAlmostEqual(observed, 0.3162, 4)


class TestScumble(unittest.TestCase):
    def test_with_nan(self):
        labels = np.array([
            [1, 0, 0, 1],
            [0, 1, 1, 0],
            [1, 0, 1, 0],
            [np.nan, np.nan, np.nan, np.nan],
            [1, 0, 0, 1],
            [0, 1, 0, 1],
            [np.nan, np.nan, 0, 1],
            [0, 1, np.nan, np.nan]
        ]).astype(float)
        self.assertAlmostEqual(scumble(labels), 0.01016, 4)


class TestAdjustedHammingDistance(unittest.TestCase):
    def test_neighbors_with_no_nan(self):
        sample = np.array([1., 0., 1.])
        neighbors = np.array([[1., 0., 1.], [0., 1., 0.], [0., 1., 1.]])
        dist = adjusted_hamming_distance(sample, neighbors)
        self.assertListEqual(dist.round(2).tolist(), [0., 1., 0.5])

    def test_neighbors_with_nan(self):
        sample = np.array([1., 0., 1.])
        neighbors = np.array([[1., 0., 1.], [0., 1., 0.], [np.nan, np.nan, 1.]])
        dist = adjusted_hamming_distance(sample, neighbors)
        self.assertListEqual(dist.tolist(), [0., 1., 0.])

    def test_sample_with_nan(self):
        sample = np.array([1., 0., np.nan])
        neighbors = np.array([[1., 0., 1.], [0., 1., 0.], [np.nan, np.nan, 1.]])
        dist = adjusted_hamming_distance(sample, neighbors)
        self.assertListEqual(dist.tolist(), [0., 1., 0.])

    def test_bad_dummy_coding(self):
        sample = np.array([2., 0., np.nan])
        neighbors = np.array([[1., 0., 1.], [0., 1., 0.]])
        self.assertRaises(ValueError, adjusted_hamming_distance, sample, neighbors)

    def test_not_dummy_coded(self):
        sample = np.array([12., 0., 1.])
        neighbors = np.array([[1., 0., -1.], [0., 1., 0.], [np.nan, np.nan, 1.], [1., 0., 0.]])
        self.assertRaises(ValueError, adjusted_hamming_distance, sample, neighbors)

    def test_difference_sizes_shape(self):
        sample = np.array([12., 0.])
        neighbors = np.array([[1., 0., -1.], [0., 1., 0.], [np.nan, np.nan, 1.], [1., 0., 0.]])
        self.assertRaises(ValueError, adjusted_hamming_distance, sample, neighbors)

    def test_incorrect_shape(self):
        sample = np.array([[12., 0., 1.]])
        neighbors = np.array([[1., 0., -1.], [0., 1., 0.], [np.nan, np.nan, 1.], [1., 0., 0.]])
        self.assertRaises(ValueError, adjusted_hamming_distance, sample, neighbors)


class TestImbalanceRatio(unittest.TestCase):
    def test_sample_sample_size(self):
        sample_sizes = np.array([10, 10])
        ir = imbalance_ratio(sample_sizes)
        self.assertTrue((ir == 1.).all())

    def test_unequal_sample_size(self):
        sample_sizes = np.array([5, 10])
        ir = imbalance_ratio(sample_sizes).tolist()
        expectation = [2., 1.]
        self.assertListEqual(ir, expectation)


class TestMLROS(unittest.TestCase):
    def test_simple_balance(self):
        labels = pd.DataFrame([[1., 0.], [1., 0.], [1., 0.], [0., 1.]], columns=['Gender_Majority', 'Gender_Minority'])
        resampler = MLROS(max_clone_percentage=0.5, random_state=1, sample_size_chunk=1)
        new_data = resampler.balance(labels=labels)
        expected = np.array([[1., 0.],
                             [1., 0.],
                             [1., 0.],
                             [0., 1.],
                             [0., 1.],
                             [0., 1.]])
        self.assertTrue(np.array_equal(new_data['labels'].values, expected))


class TestMLRUS(unittest.TestCase):
    def test_simple_balance(self):
        labels = pd.DataFrame([[1., 0.], [1., 0.], [1., 0.], [0., 1.]], columns=['Gender_Majority', 'Gender_Minority'])
        resampler = MLRUS(random_state=1, sample_size_chunk=1)
        new_data = resampler.balance(labels=labels)
        expected = np.array([[1., 0.],
                             [0., 1.]])
        self.assertTrue(np.array_equal(new_data['labels'].values, expected))


class TestMLSMOTE(unittest.TestCase):
    def test_simple_balance(self):
        labels = pd.DataFrame([[1., 0.], [1., 0.], [1., 0.], [0., 1.], [0., 1.]], columns=['Gender_Majority', 'Gender_Minority'])
        features = pd.DataFrame([[0.5, 0.2], [0.2, 0.1], [0., 0.8], [0.1, 0.1], [1., 1.2]], columns=['feature_a', 'feature_b'])
        resampler = MLSMOTE(random_state=1, k_neighbors=1)
        new_data = resampler.balance(labels=labels, features=features)
        expected = np.array([[0., 1.],
                             [1., 0.],
                             [1., 0.],
                             [1., 0.],
                             [0., 1.],
                             [0., 1.]])
        self.assertTrue(np.array_equal(new_data['labels'].values, expected))


class TestMLeNN(unittest.TestCase):
    def test_simple_balance(self):
        labels = pd.DataFrame([[1., 0.], [1., 0.], [1., 0.], [0., 1.], [0., 1.]], columns=['Gender_Majority', 'Gender_Minority'])
        features = pd.DataFrame([[0.5, 0.2], [0.2, 0.1], [0., 0.8], [0.1, 0.1], [1., 1.2]], columns=['feature_a', 'feature_b'])
        resampler = MLeNN(random_state=1, k_neighbors=1)
        new_data = resampler.balance(labels=labels, features=features)
        expected = np.array([[1., 0.],
                             [1., 0.],
                             [0., 1.],
                             [0., 1.]])
        self.assertTrue(np.array_equal(new_data['labels'].values, expected))
