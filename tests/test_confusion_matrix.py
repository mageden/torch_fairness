import unittest

import torch

from torch_fairness.confusion_matrix import true_negative
from torch_fairness.confusion_matrix import false_positive
from torch_fairness.confusion_matrix import true_positive
from torch_fairness.confusion_matrix import false_negative
from torch_fairness.confusion_matrix import true_negative_rate
from torch_fairness.confusion_matrix import false_positive_rate
from torch_fairness.confusion_matrix import true_positive_rate
from torch_fairness.confusion_matrix import false_negative_rate
from torch_fairness.confusion_matrix import ConfusionMatrix
from torch_fairness.confusion_matrix import _validate_input


class TestConfusionMatrix(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.labels = torch.tensor([0, 0, 1, 0, 1, 0, 1, 0, 1, 1])
        cls.pred = torch.tensor([1, 1, 1, 1, 1, 0, 0, 1, 0, 0])
        cls.cm = ConfusionMatrix(["tpr", "fpr", "tnr", "fnr", "ppv", "npv", "accuracy"])

    @classmethod
    def tearDownClass(cls):
        cls.labels = None
        cls.pred = None

    def test_confusion_matrix_class(self):
        matrix = self.cm.calculate(labels=self.labels, pred=self.pred)
        matrix = {key: value.item() for key, value in matrix.items()}
        predicted = {
            "tpr": 0.4,
            "fpr": 0.8,
            "tnr": 0.2,
            "fnr": 0.6,
            "ppv": 0.3333333333333333,
            "npv": 0.25,
            "accuracy": 0.3,
        }
        for key in predicted.keys():
            self.assertAlmostEqual(
                matrix[key], predicted[key], places=4, msg=f"{key} is incorrect"
            )

    def test_no_metric_provided(self):
        self.assertRaises(ValueError, ConfusionMatrix, metrics=None)


class TestConfusionMatrixMetrics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.labels = torch.tensor([0, 0, 1, 0, 1, 0, 1, 0, 1, 1])
        cls.pred = torch.tensor([1, 1, 1, 1, 1, 0, 0, 1, 0, 0])

    @classmethod
    def tearDownClass(cls):
        cls.labels = None
        cls.pred = None

    def test_true_positive(self):
        tp = true_positive(self.labels, self.pred)
        self.assertEqual(tp, torch.tensor(2))

    def test_false_positive(self):
        tp = false_positive(self.labels, self.pred)
        self.assertEqual(tp, torch.tensor(4))

    def test_true_negative(self):
        tp = true_negative(self.labels, self.pred)
        self.assertEqual(tp, torch.tensor(1))

    def test_false_negative(self):
        tp = false_negative(self.labels, self.pred)
        self.assertEqual(tp, torch.tensor(3))

    def test_true_positive_rate(self):
        tpr = true_positive_rate(labels=self.labels, pred=self.pred)
        self.assertEqual(tpr, torch.tensor(0.4))

    def test_false_positive_rate(self):
        tpr = false_positive_rate(labels=self.labels, pred=self.pred)
        self.assertEqual(tpr, torch.tensor(0.8))

    def test_true_negative_rate(self):
        tpr = true_negative_rate(labels=self.labels, pred=self.pred)
        self.assertEqual(tpr, torch.tensor(0.2))

    def test_false_negative_rate(self):
        tpr = false_negative_rate(labels=self.labels, pred=self.pred)
        self.assertEqual(tpr, torch.tensor(0.6))

    def test_positive_predictive_value(self):
        tpr = false_negative_rate(labels=self.labels, pred=self.pred)
        self.assertEqual(tpr, torch.tensor(0.6))

    def test_negative_predictive_value(self):
        tpr = false_negative_rate(labels=self.labels, pred=self.pred)
        self.assertEqual(tpr, torch.tensor(0.6))

    def test_unequal_shapes(self):
        self.assertRaises(
            ValueError, _validate_input, torch.tensor([1, 2]), torch.tensor([2])
        )

    def test_incorrect_dim(self):
        self.assertRaises(
            ValueError, _validate_input, torch.rand(2, 1, 3), torch.rand(2, 1, 3)
        )
