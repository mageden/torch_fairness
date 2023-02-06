import unittest

import torch
import numpy as np

from torch_fairness.data import SensitiveAttribute
from torch_fairness.data import SensitiveMap

from torch_fairness.metrics import Threshold
from torch_fairness.metrics import MinimaxFairness
from torch_fairness.metrics import FalsePositiveRateBalance
from torch_fairness.metrics import EqualOpportunity
from torch_fairness.metrics import EqualizedOdds
from torch_fairness.metrics import AccuracyEquality
from torch_fairness.metrics import PredictiveParity
from torch_fairness.metrics import ConditionalUseAccuracyParity
from torch_fairness.metrics import BalancedPositive
from torch_fairness.metrics import BalancedNegative
from torch_fairness.metrics import DemographicParity
from torch_fairness.metrics import SmoothedEmpiricalDifferentialFairness
from torch_fairness.metrics import WeightedSumofLogs
from torch_fairness.metrics import MeanDifferences
from torch_fairness.metrics import CrossPairDistance
from torch_fairness.metrics import MMDFairness
from torch_fairness.metrics import AbsoluteCorrelation


class TestThreshold(unittest.TestCase):
    def test_soft_threshold_value(self):
        x = torch.tensor([-1., 0., 1.], requires_grad=True)
        observed = Threshold(threshold=0., use_hard_threshold=False, alpha=25.)(x).detach().numpy().tolist()
        expected = [0.0000, 0.5000, 1.0000]
        for i, j in zip(observed, expected):
            self.assertAlmostEqual(i, j, 4)

    def test_hard_threshold_value(self):
        x = torch.tensor([-1., 0., 1.], requires_grad=True)
        observed = Threshold(threshold=0., use_hard_threshold=True)(x).detach().numpy().tolist()
        expected = [0., 0., 1.]
        for i, j in zip(observed, expected):
            self.assertAlmostEqual(i, j, 4)

    def test_hard_threshold_grad(self):
        x = torch.tensor([0.], requires_grad=True)
        observed = Threshold(threshold=0., use_hard_threshold=True)(x)
        self.assertTrue(observed.requires_grad)


class TestConditionalUseAccuracyParity(unittest.TestCase):
    def setUp(self) -> None:
        sensitive_map = SensitiveMap(
            SensitiveAttribute(majority=0, minority=[1], name="TestSensitive")
        )
        self.metric = ConditionalUseAccuracyParity(sensitive_map=sensitive_map, use_hard_threshold=True)

    def tearDown(self) -> None:
        self.metric = None

    def test_expected(self):
        pred = torch.tensor([0.0, 0.0, 1.0, 0.0, 1.0, 0.0]).unsqueeze(1)
        sensitive = torch.tensor(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ]
        )
        labels = torch.tensor([1.0, 1.0, 0.0, 1.0, 1.0, 1.0]).unsqueeze(1)
        observed = self.metric(pred=pred, sensitive=sensitive, labels=labels)
        self.assertAlmostEqual(observed.item(), 0.5, 1)


class TestPredictiveParity(unittest.TestCase):
    def setUp(self) -> None:
        sensitive_map = SensitiveMap(
            SensitiveAttribute(majority=0, minority=[1], name="TestSensitive")
        )
        self.metric = PredictiveParity(sensitive_map=sensitive_map, use_hard_threshold=True)

    def tearDown(self) -> None:
        self.metric = None

    def test_expected(self):
        pred = torch.tensor([0.0, 0.0, 1.0, 0.0, 1.0, 1.0]).unsqueeze(1)
        sensitive = torch.tensor(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ]
        )
        labels = torch.tensor([1.0, 1.0, 0.0, 1.0, 1.0, 0.0]).unsqueeze(1)
        observed = self.metric(pred=pred, sensitive=sensitive, labels=labels)
        self.assertAlmostEqual(observed.item(), 0.5, 1)


class TestAccuracyEquality(unittest.TestCase):
    def setUp(self) -> None:
        sensitive_map = SensitiveMap(
            SensitiveAttribute(majority=0, minority=[1], name="TestSensitive")
        )
        self.metric = AccuracyEquality(sensitive_map=sensitive_map, use_hard_threshold=True)

    def tearDown(self) -> None:
        self.metric = None

    def test_expected(self):
        pred = torch.tensor([0.0, 0.0, 1.0, 0.0, 1.0, 1.0]).unsqueeze(1)
        sensitive = torch.tensor(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ]
        )
        labels = torch.tensor([1.0, 1.0, 0.0, 1.0, 1.0, 0.0]).unsqueeze(1)
        observed = self.metric(pred=pred, sensitive=sensitive, labels=labels)
        self.assertAlmostEqual(observed.item(), 0.33, 2)


class TestEqualizedOdds(unittest.TestCase):
    def setUp(self) -> None:
        sensitive_map = SensitiveMap(
            SensitiveAttribute(majority=0, minority=[1], name="TestSensitive")
        )
        self.metric = EqualizedOdds(sensitive_map=sensitive_map, use_hard_threshold=True)

    def tearDown(self) -> None:
        self.metric = None

    def test_expected(self):
        pred = torch.tensor([0.0, 0.0, 1.0, 0.0, 1.0, 1.0]).unsqueeze(1)
        sensitive = torch.tensor(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ]
        )
        labels = torch.tensor([1.0, 1.0, 0.0, 1.0, 1.0, 0.0]).unsqueeze(1)
        observed = self.metric(pred=pred, sensitive=sensitive, labels=labels)
        self.assertAlmostEqual(observed.item(), 0.25, 2)


class TestEqualOpportunity(unittest.TestCase):
    def setUp(self) -> None:
        sensitive_map = SensitiveMap(
            SensitiveAttribute(majority=0, minority=[1], name="TestSensitive")
        )
        self.metric = EqualOpportunity(sensitive_map=sensitive_map, use_hard_threshold=True)

    def tearDown(self) -> None:
        self.metric = None

    def test_expected(self):
        pred = torch.tensor([0.0, 0.0, 1.0, 0.0, 1.0, 0.0]).unsqueeze(1)
        sensitive = torch.tensor(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ]
        )
        labels = torch.tensor([1.0, 1.0, 0.0, 1.0, 1.0, 0.0]).unsqueeze(1)
        observed = self.metric(pred=pred, sensitive=sensitive, labels=labels)
        self.assertAlmostEqual(observed.item(), 0.5, 1)


class TestFalsePositiveRateBalance(unittest.TestCase):
    def setUp(self) -> None:
        sensitive_map = SensitiveMap(
            SensitiveAttribute(majority=0, minority=[1], name="TestSensitive")
        )
        self.metric = FalsePositiveRateBalance(sensitive_map=sensitive_map, use_hard_threshold=True)

    def tearDown(self) -> None:
        self.metric = None

    def test_expected(self):
        pred = torch.tensor([1.0, 1.0, 0.0, 1.0, 0.0, 1.0]).unsqueeze(1)
        sensitive = torch.tensor(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ]
        )
        labels = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 1.0]).unsqueeze(1)
        observed = self.metric(pred=pred, sensitive=sensitive, labels=labels)
        self.assertAlmostEqual(observed.item(), 0.5, 1)


class TestMinimaxFairness(unittest.TestCase):
    @classmethod
    def setUp(cls) -> None:
        sensitive_map = SensitiveMap(
            SensitiveAttribute(minority=[1], majority=0, name="tests")
        )
        cls.dp = MinimaxFairness(sensitive_map=sensitive_map)

    @classmethod
    def tearDown(cls) -> None:
        cls.sensitive_map = None

    def test_expected(self):
        losses = torch.tensor([0.9, 0.1, 0.8, 0.95])
        sensitive = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]])
        observed = self.dp(losses, sensitive)
        self.assertAlmostEqual(observed.item(), 0.883, 3)

    def test_incorrect_size(self):
        losses = torch.tensor([[0.9, 0.1, 0.8, 0.95]])
        sensitive = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]])
        self.assertRaises(ValueError, self.dp, loss=losses, sensitive=sensitive)


class TestDemographicParity(unittest.TestCase):
    @classmethod
    def setUp(cls) -> None:
        cls.sensitive_map = SensitiveMap(
            SensitiveAttribute(minority=[1], majority=0, name="tests")
        )

    @classmethod
    def tearDown(cls) -> None:
        cls.sensitive_map = None

    def test_selection_rate(self):
        pred = torch.tensor([0, 1]).unsqueeze(1)
        dp = DemographicParity(sensitive_map=self.sensitive_map, threshold=None)
        result = dp.selection_rate(pred)
        self.assertEqual(result.item(), torch.tensor(0.5))

    def test_threshold(self):
        pred = torch.tensor([-1, 1, -1, -1]).unsqueeze(1)
        sensitive = torch.tensor([[1, 1, 0, 0], [0, 0, 1, 1]]).T
        dp = DemographicParity(sensitive_map=self.sensitive_map, threshold=0, use_hard_threshold=True)
        result = dp.forward(pred, sensitive)
        self.assertEqual(result.item(), torch.tensor(0.5))


class TestSmoothedEmpiricalDifferentialFairness(unittest.TestCase):
    @classmethod
    def setUp(cls) -> None:
        cls.sensitive_map = SensitiveMap(
            SensitiveAttribute(minority=[1], majority=0, name="tests")
        )
        cls.sedf = SmoothedEmpiricalDifferentialFairness(
            sensitive_map=cls.sensitive_map, use_hard_threshold=True
        )

    @classmethod
    def tearDown(cls) -> None:
        cls.sedf = None

    def test_smoothed_selection_rate(self):
        result = self.sedf._smooth_hire_prop(0, 0)
        self.assertEqual(result, 0.5)

        result = self.sedf._smooth_hire_prop(0, 1)
        self.assertEqual(result, 0.25)

    def test_no_difference(self):
        pred = torch.tensor([0, 1, 0, 1]).unsqueeze(1)
        sensitive = torch.tensor([[1, 1, 0, 0], [0, 0, 1, 1]]).T
        result = self.sedf.forward(pred=pred, sensitive=sensitive)
        self.assertEqual(result.item(), 0.0)

    def test_difference(self):
        pred = torch.tensor([1, 1, 1, 1, 1, 0, 1, 1, 1, 1]).unsqueeze(1)
        sensitive = torch.tensor(
            [[1, 1, 1, 1, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]]
        ).T
        result = self.sedf.forward(pred=pred, sensitive=sensitive)
        self.assertAlmostEqual(result.item(), 1.0986, 4)

    def test_order_invariant(self):
        pred = torch.tensor([1, 1, 1, 1, 1, 0, 1, 1, 1, 1]).unsqueeze(1)
        sensitive = torch.tensor(
            [[1, 1, 1, 1, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]]
        ).T
        result = self.sedf.forward(pred=pred, sensitive=sensitive)
        result_reverse = self.sedf.forward(pred=pred, sensitive=1 - sensitive)
        self.assertAlmostEqual(result.item(), result_reverse.item(), 4)


class TestBalancedPositive(unittest.TestCase):
    @classmethod
    def setUp(cls):
        cls.sensitive_map = SensitiveMap(
            SensitiveAttribute(minority=[1], majority=0, name="tests")
        )
        cls.bp = BalancedPositive(sensitive_map=cls.sensitive_map, use_hard_threshold=True)

    @classmethod
    def tearDown(cls):
        cls.bp = None

    def test_no_difference(self):
        sensitive = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ).T
        labels = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]).unsqueeze(1)
        pred = torch.tensor([-99.0, 0.0, -99.0, 0.0, 99.0, 0.0, 99.0, 0.0]).unsqueeze(1)
        out = self.bp.forward(pred=pred, labels=labels, sensitive=sensitive).item()
        self.assertAlmostEqual(out, 0.0, 4)

    def test_difference(self):
        sensitive = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ).T
        labels = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]).unsqueeze(1)
        pred = torch.tensor([-99.0, 1.0, -99.0, 1.0, 99.0, 0.0, 99.0, 0.0]).unsqueeze(1)
        out = self.bp.forward(pred=pred, labels=labels, sensitive=sensitive).item()
        self.assertAlmostEqual(out, 1.0, 4)


class TestBalancedNegative(unittest.TestCase):
    @classmethod
    def setUp(cls):
        cls.sensitive_map = SensitiveMap(
            SensitiveAttribute(minority=[1], majority=0, name="tests")
        )
        cls.bn = BalancedNegative(sensitive_map=cls.sensitive_map, use_hard_threshold=True)

    @classmethod
    def tearDown(cls):
        cls.bn = None

    def test_no_difference(self):
        sensitive = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ).T
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]).unsqueeze(1)
        pred = torch.tensor([-99.0, 0.0, -99.0, 0.0, 99.0, 0.0, 99.0, 0.0]).unsqueeze(1)
        out = self.bn.forward(pred=pred, labels=labels, sensitive=sensitive).item()
        self.assertAlmostEqual(out, 0.0, 4)

    def test_difference(self):
        sensitive = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ).T
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]).unsqueeze(1)
        pred = torch.tensor([-99.0, 1.0, -99.0, 1.0, 99.0, 0.0, 99.0, 0.0]).unsqueeze(1)
        out = self.bn.forward(pred=pred, labels=labels, sensitive=sensitive).item()
        self.assertAlmostEqual(out, 1.0, 4)


class TestMeanDifferences(unittest.TestCase):
    @classmethod
    def setUp(cls):
        cls.sensitive_map = SensitiveMap(
            SensitiveAttribute(minority=[1], majority=0, name="tests")
        )
        cls.md = MeanDifferences(sensitive_map=cls.sensitive_map)

    @classmethod
    def tearDown(cls):
        cls.md = None

    def test_no_difference(self):
        majority = torch.tensor([-1.0, 1.0])
        minority = torch.tensor([-1.0, 1.0])
        result = self.md.calculate(majority, minority)
        self.assertEqual(result.item(), 0.0)

    def test_difference(self):
        majority = torch.tensor([-1.0, -1.0])
        minority = torch.tensor([1.0, 1.0])
        result = self.md.calculate(majority, minority)
        self.assertEqual(result.item(), 2.0)

    def test_expected(self):
        pred = torch.tensor([0.0, 2.0, -1.0, 1.0]).unsqueeze(1)
        sensitive = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
        observed = self.md(pred=pred, sensitive=sensitive)
        self.assertAlmostEqual(observed, 1.0, 1)


class TestWeightedSumofLogs(unittest.TestCase):
    @classmethod
    def setUp(cls):
        cls.sensitive_map = SensitiveMap(
            SensitiveAttribute(minority=[1], majority=0, name="tests")
        )

    @classmethod
    def tearDown(cls):
        cls.sensitive_map = None

    def test_even_historical_bias(self):
        sensitive = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0, 0, 0]]).T
        labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1]).unsqueeze(1)
        wsl = WeightedSumofLogs(
            sensitive_map=self.sensitive_map, sensitive=sensitive, labels=labels
        )
        result = wsl.historical_bias.squeeze().numpy().tolist()
        self.assertListEqual(result, [0.125, 0.125])

    def test_uneven_historical_bias(self):
        sensitive = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0, 0, 0]]).T
        labels = torch.tensor([0, 1, 1, 1, 0, 0, 0, 1]).unsqueeze(1)
        wsl = WeightedSumofLogs(
            sensitive_map=self.sensitive_map, sensitive=sensitive, labels=labels
        )
        result = wsl.historical_bias.squeeze().numpy().tolist()
        self.assertListEqual(result, [0.1875, 0.0625])

    def test_expected(self):
        sensitive = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1]])
        pred = torch.tensor([0.7, 0.6, 0.6, 0.2])
        labels = torch.tensor([1, 0, 0, 1])
        sensitive_map = SensitiveMap(
            SensitiveAttribute(name="tests", majority=0, minority=[1])
        )
        fair_loss = WeightedSumofLogs(
            labels=labels, sensitive=sensitive, sensitive_map=sensitive_map
        )
        overall_loss = fair_loss(pred=pred, sensitive=sensitive).mean()
        self.assertAlmostEqual(overall_loss.item(), 0.0934, 4)


class TestCrossPairDistance(unittest.TestCase):
    @classmethod
    def setUp(cls):
        cls.sensitive_map = SensitiveMap(
            SensitiveAttribute(minority=[1], majority=0, name="tests")
        )
        cls.metric = CrossPairDistance(
            sensitive_map=cls.sensitive_map, is_regression=True
        )

    @classmethod
    def tearDown(cls):
        cls.sensitive_map = None

    def test_pairwise_distance_shape(self):
        x = torch.tensor([0, 0, 0])
        y = torch.tensor([0, 0])
        out = self.metric.pairwise_difference(x, y)
        self.assertEqual(list(out.shape), [3, 2])

    def test_pairwise_distance(self):
        x = torch.tensor([1])
        y = torch.tensor([-1, 0, 1])
        out = self.metric.pairwise_difference(x, y)
        self.assertListEqual(out.squeeze().numpy().tolist(), [2, 1, 0])

    def test_label_pairwise_distance_shape(self):
        x = torch.tensor([0, 0, 0])
        y = torch.tensor([0, 0])
        out = self.metric.pairwise_label_distance(x, y)
        self.assertEqual(list(out.shape), [3, 2])

    def test_regression_label_pairwise_distance(self):
        x = torch.tensor([1])
        y = torch.tensor([-1, 0, 1])
        out = self.metric.pairwise_label_distance(x, y)
        expected = [0.0183, 0.3679, 1.0000]
        for observed_i, expected_i in zip(out.squeeze().numpy().tolist(), expected):
            self.assertAlmostEqual(observed_i, expected_i, 4)

    def test_classification_label_pairwise_distance(self):
        x = torch.tensor([1])
        y = torch.tensor([1, 0, 1])
        cpd = CrossPairDistance(sensitive_map=self.sensitive_map, is_regression=False)
        expected = [1.0, 0.0, 1.0]
        out = cpd.pairwise_label_distance(x, y)
        for observed_i, expected_i in zip(out.squeeze().numpy().tolist(), expected):
            self.assertAlmostEqual(observed_i, expected_i, 4)

    def test_identical(self):
        pred = torch.tensor([0.5, 0.25, 0.5, 0.25]).unsqueeze(1)
        sensitive = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
        labels = torch.tensor([0, 1, 0, 1]).unsqueeze(1)
        observed = self.metric(pred=pred, sensitive=sensitive, labels=labels)
        self.assertAlmostEqual(observed.item(), 0.0, 3)

    def test_difference(self):
        pred = torch.tensor([0.5, 0.5, 0.25, 0.25]).unsqueeze(1)
        sensitive = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
        labels = torch.tensor([0, 1, -1, 1]).unsqueeze(1)
        observed = self.metric(pred=pred, sensitive=sensitive, labels=labels)
        self.assertAlmostEqual(observed.item(), 0.012, 3)

    def test_size_mismatch(self):
        pred = torch.tensor([[1, 0, 1]]).unsqueeze(1)
        labels = torch.tensor([1, 0, 1]).unsqueeze(1)
        sensitive = torch.tensor([[0.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
        self.assertRaises(
            ValueError, self.metric, pred=pred, labels=labels, sensitive=sensitive
        )


class TestAbsoluteCorrelation(unittest.TestCase):
    @classmethod
    def setUp(cls):
        cls.sensitive_map = SensitiveMap(
            SensitiveAttribute(minority=[1], majority=0, name="tests")
        )
        cls.ac = AbsoluteCorrelation(sensitive_map=cls.sensitive_map)

    @classmethod
    def tearDown(cls):
        cls.ac = None

    def test_no_correlation(self):
        minority = torch.tensor([1.0 for _ in range(10)])
        majority = torch.tensor([1.0 for _ in range(10)])
        absolute_correlation = self.ac.calculate(minority, majority).item()
        self.assertAlmostEqual(absolute_correlation, 0.0, 4)

    def test_perfect_correlation(self):
        minority = torch.normal(0, 1, (100,))
        majority = minority.clone()
        absolute_correlation = self.ac.calculate(minority, majority).item()
        self.assertAlmostEqual(absolute_correlation, 1.0, 4)

    def test_negative_correlation(self):
        minority = torch.normal(0, 1, (100,))
        majority = -minority.clone()
        absolute_correlation = self.ac.calculate(minority, majority).item()
        self.assertAlmostEqual(absolute_correlation, 1.0, 4)

    def test_partial_correlation(self):
        minority = torch.tensor([1.7366, 0.2834, 0.4598, -0.1728, -1.0838])
        majority = torch.tensor([0.0362, -0.3253, 0.9198, 0.4857, -1.6386])
        absolute_correlation = self.ac.calculate(minority, majority).item()
        self.assertAlmostEqual(absolute_correlation, 0.551777809078049, 4)

    def test_expectation(self):
        pred = torch.tensor([0.25, 0.5, 0.75, 1.0]).unsqueeze(1)
        sensitive = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
        observed = self.ac(pred=pred, sensitive=sensitive)
        self.assertAlmostEqual(observed.item(), 0.8944, 4)


class TestMMDFairness(unittest.TestCase):
    @classmethod
    def setUp(cls):
        cls.sensitive_map = SensitiveMap(
            SensitiveAttribute(minority=[1], majority=0, name="tests")
        )
        cls.mmd = MMDFairness(sensitive_map=cls.sensitive_map)

    @classmethod
    def tearDown(cls):
        cls.mmd = None

    def test_gaussian_kernel(self):
        self.mmd.bandwidth = 1.0
        result = self.mmd.gaussian_kernel(torch.tensor([2])).item()
        self.assertAlmostEqual(result, 0.0183, 3)

    def test_identical_distribution(self):
        group_a = np.random.normal(0, 1, (5, 2))
        group_b = group_a.copy()
        np.random.shuffle(group_b)
        group_a = torch.tensor(group_a)
        group_b = torch.tensor(group_b)
        result = self.mmd.calculate(group_a, group_b).item()
        self.assertAlmostEqual(result, 0, places=3)

    def test_symmetric(self):
        group_a = np.random.normal(0, 1, (5, 2))
        group_b = np.random.normal(0, 1, (5, 2))
        group_a = torch.tensor(group_a)
        group_b = torch.tensor(group_b)
        result_a = self.mmd.calculate(group_a, group_b).item()
        result_b = self.mmd.calculate(group_b, group_a).item()
        self.assertAlmostEqual(result_a, result_b, places=3)
