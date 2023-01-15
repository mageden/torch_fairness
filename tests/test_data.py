import unittest

import numpy as np
import torch

from torch_fairness.exceptions import NotFittedError
from torch_fairness.exceptions import DummyCodingError
from torch_fairness.data import SensitiveMap
from torch_fairness.data import SensitiveTransformer
from torch_fairness.data import SensitiveAttribute
from torch_fairness.data import get_member


class TestGetMember(unittest.TestCase):
    def test_dimension_error(self):
        sensitive = torch.tensor(np.random.random((3, 2, 3)))
        x = torch.tensor(np.random.random((3, 2, 3)))
        self.assertRaises(ValueError, get_member, sensitive=sensitive, x=x)

    def test_mismatched_sizes(self):
        sensitive = torch.tensor(np.random.random((4,)))
        x = torch.tensor(np.random.random((3,)))
        self.assertRaises(ValueError, get_member, sensitive=sensitive, x=x)


class TestSensitiveMap(unittest.TestCase):
    def test_invalid_input(self):
        self.assertRaises(ValueError, SensitiveMap, [])

    def test_indexing(self):
        attribute = {"name": "Gender", "majority": 0, "minority": [1, 2]}
        sensitive_map = SensitiveMap(attribute)
        attribute = SensitiveAttribute(**attribute)
        self.assertEqual(attribute, sensitive_map[0])

    def test_infer(self):
        data = np.array([[1], [1], [2], [2], [1], [4]])
        inferred = SensitiveMap.infer(data, minimum_sample_size=2)
        expected = SensitiveMap(SensitiveAttribute(name=0, majority=1, minority=[2]))
        self.assertEqual(inferred, expected)

    def test_insufficient_groups(self):
        data = np.array([[1], [2]])
        self.assertRaises(ValueError, SensitiveMap.infer, data, minimum_sample_size=2)

    def test_incompatable_comparison(self):
        object_a = SensitiveMap(SensitiveAttribute(name=1, majority=1, minority=[2]))
        object_b = SensitiveAttribute(name=1, majority=1, minority=[2])
        self.assertNotEqual(object_a, object_b)

    def test_incorrect_input(self):
        x = torch.tensor(np.random.random((10, 1)))
        self.assertRaises(ValueError, SensitiveMap.infer, x=x, minimum_sample_size=2)


class TestSensitiveTransformer(unittest.TestCase):
    def setUp(self) -> None:
        self.x = np.array([[0], [0], [1], [1], [2]])

    def tearDown(self) -> None:
        self.x = None

    def test_sample_size_input_error(self):
        self.assertRaises(ValueError, SensitiveTransformer, minimum_sample_size=-1)

    def test_sensitive_map_input_error(self):
        self.assertRaises(
            ValueError,
            SensitiveTransformer,
            sensitive_map={"name": "Gender", "minority": [1], "majority": 0},
        )

    def test_insufficient_sample_size(self):
        sensitive_map = SensitiveMap(
            SensitiveAttribute(name="Gender", minority=[1, 2], majority=0)
        )
        transformer = SensitiveTransformer(
            minimum_sample_size=5, sensitive_map=sensitive_map
        )
        self.assertRaises(ValueError, transformer.fit, x=self.x)

    def test_missing_groups(self):
        sensitive_map = SensitiveMap(
            SensitiveAttribute(name="Gender", minority=[3], majority=0)
        )
        transformer = SensitiveTransformer(
            minimum_sample_size=1, sensitive_map=sensitive_map
        )
        self.assertRaises(ValueError, transformer.fit, x=self.x)

    def test_infer(self):
        transformer = SensitiveTransformer(minimum_sample_size=1)
        transformer.fit(self.x)
        expected = SensitiveMap(SensitiveAttribute(name=0, majority=0, minority=[1, 2]))
        self.assertEqual(expected, transformer.sensitive_map)

    def test_recreation(self):
        sensitive_map = SensitiveMap(
            SensitiveAttribute(name=0, majority=0, minority=[1, 2])
        )
        transformer = SensitiveTransformer(
            minimum_sample_size=1, sensitive_map=sensitive_map
        )
        x_new = transformer.fit_transform(self.x)
        x_recreate = transformer.inverse_transform(x_new)
        self.assertListEqual(
            self.x.squeeze().astype(float).tolist(),
            x_recreate.squeeze().astype(float).tolist(),
        )

    def test_error_if_not_trained(self):
        sensitive_map = SensitiveMap(
            SensitiveAttribute(name=0, majority=0, minority=[1, 2])
        )
        transformer = SensitiveTransformer(
            minimum_sample_size=1, sensitive_map=sensitive_map
        )
        x_new = np.array(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        self.assertRaises(NotFittedError, transformer.inverse_transform, x=x_new)

    def test_dummy_coding_error(self):
        sensitive_map = SensitiveMap(
            SensitiveAttribute(name=0, majority=0, minority=[1, 2])
        )
        transformer = SensitiveTransformer(
            minimum_sample_size=1, sensitive_map=sensitive_map
        )
        transformer.fit(self.x)
        x_new = np.array(
            [
                [2.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 3.3],
                [0.0, 0.0, 1.0],
            ]
        )
        self.assertRaises(DummyCodingError, transformer.inverse_transform, x=x_new)
