import math
from abc import ABC
from typing import Callable, List, Literal, Optional

import torch
import torch.nn.functional as F

from torch_fairness.confusion_matrix import ConfusionMatrix
from torch_fairness.data import SensitiveMap, get_member


def _validate_shape(x: torch.tensor, squeeze: bool = True):
    """
    Validates and standardizes shape.

    Parameters
    ----------

    x: torch.tensor
        Tensor to be validated

    squeeze: bool = True
        If True then ensure output tensor has ndim=1, otherwise ensure ndim=2 with shape[1]==1

    Returns
    -------

    out: torch.tensor
        Validated/standardize tensor
    """
    if x.ndim > 2:
        raise ValueError()
    elif x.ndim == 2:
        if x.shape[1] == 1:
            if squeeze:
                x = x.squeeze()
        else:
            raise ValueError()
    elif x.ndim == 1 and not squeeze:
        x = x.unsqueeze(1)
    return x


class ThresholdOperator(torch.nn.Module):
    def __init__(self, threshold: Optional[float] = None):
        super().__init__()
        self.threshold = threshold

    def forward(self, x: torch.tensor) -> torch.tensor:
        if self.threshold is not None:
            x = (x >= self.threshold) * 1.0
        return x


class BaseFairnessLoss(ABC, torch.nn.Module):
    def __init__(self, sensitive_map: SensitiveMap):
        super().__init__()
        self.sensitive_map = sensitive_map


class ConfusionMatrixFairness(BaseFairnessLoss):
    supports = ["classification"]

    def __init__(
        self,
        cm_metrics: list[Literal["tpr", "fpr", "tnr", "fnr", "ppv", "npv", "accuracy"]],
        threshold: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.threshold_layer = ThresholdOperator(threshold=threshold)
        self.cm = ConfusionMatrix(cm_metrics)

    def forward(
        self, pred: torch.tensor, labels: torch.tensor, sensitive: torch.tensor
    ) -> torch.tensor:
        """
        Parameters
        ----------
        pred: torch.tensor
            Predictions.

        sensitive: torch.tensor
            Dummy-coded minority/majority sensitive groups.

        labels: torch.tensor
            Criterion labels.

        Returns
        -------
        Group fairness loss: torch.tensor
        """
        pred = _validate_shape(pred, squeeze=True)
        labels = _validate_shape(labels, squeeze=True)
        pred = self.threshold_layer(pred)
        result = []
        for maj_idx, min_groups in self.sensitive_map.majority_minority_pairs:
            maj_sensitive = sensitive[:, maj_idx]
            majority = self.cm.calculate(
                get_member(maj_sensitive, labels), get_member(maj_sensitive, pred)
            )
            majority = torch.stack(list(majority.values()))
            # Some metrics may be conditional where that condition is not observed (NPV) causing a NaN - if this happens
            # set to 0.
            majority = torch.nan_to_num(majority, 0.0)
            for min_idx in min_groups:
                sensitive_minority = sensitive[:, min_idx]
                minority = self.cm.calculate(
                    get_member(sensitive_minority, labels),
                    get_member(sensitive_minority, pred),
                )
                minority = torch.stack(list(minority.values()))
                minority = torch.nan_to_num(minority, 0.0)
                group_loss = torch.mean(torch.abs(minority - majority))
                result.append(group_loss)
        result = torch.stack(result)
        return result


class MinimaxFairness(BaseFairnessLoss):
    """Maximum loss across any of the sensitive groups [1,2].

    .. math::
        \\underset{s \\in S}{\\text{max}}(L(Y, \\hat{Y}|s))

    Parameters
    ----------
    sensitive_map: SensitiveMap
        Class containing information about the minority and majority groups - used to match minority/majority columns.

    References
    ----------
    .. [1] Martinez, N., Bertran, M., & Sapiro, G. (2020, November). Minimax pareto fairness: A multi objective
        perspective. In International Conference on Machine Learning (pp. 6755-6764). PMLR.

    .. [2] Diana, E., Gill, W., Kearns, M., Kenthapadi, K., & Roth, A. (2021, July). Minimax group fairness: Algorithms
        and experiments. In Proceedings of the 2021 AAAI/ACM Conference on AI, Ethics, and Society (pp. 66-76).

    Examples
    --------
    >>> import torch
    >>> from torch_fairness.metrics import MinimaxFairness
    >>> from torch_fairness.data import SensitiveMap, SensitiveAttribute
    >>> sensitive = torch.tensor([[1., 0.], [1., 0.], [1., 0.], [0., 1.], [0., 1], [0., 1]])
    >>> loss = torch.tensor([0.5, 0.2, 0.1, 0.99, 1.5, 2.])
    >>> sensitive_map = SensitiveMap(SensitiveAttribute(name='tests', majority=0, minority=[1]))
    >>> fair_loss = MinimaxFairness(sensitive_map=sensitive_map)
    >>> print(fair_loss(loss=loss, sensitive=sensitive))
    tensor(1.4967)
    """

    supports = ["classification", "regression"]

    def forward(self, loss: torch.tensor, sensitive: torch.tensor) -> torch.tensor:
        """
        Parameters
        ----------
        loss: torch.tensor
            Tensor of criterion losses (e.g., instance-level MSE that hasn't been reduced).

        sensitive: torch.tensor
            Tensor of dummy-coded minority/majority sensitive groups.

        Returns
        -------
        Maximum group loss: torch.tensor
        """
        loss = _validate_shape(loss, squeeze=True)
        result = []
        for col_i in range(sensitive.shape[1]):
            sensitive_i = sensitive[:, col_i]
            minority = get_member(sensitive_i, loss)
            result.append(self.calculate(loss=minority))
        result = torch.stack(result)
        result = torch.max(result)
        return result

    def calculate(self, loss: torch.tensor) -> torch.tensor:
        return loss.mean()


class FalsePositiveRateBalance(ConfusionMatrixFairness):
    """Difference in false positive rate between minority and majority groups [1].

    .. math::
        P[\\hat{Y}=1|S=0,Y=0] = P[\\hat{Y}=1|S=1,Y=0]

    Parameters
    ----------
    threshold: float = None
        Threshold used for converting prediction into label. If None then no threshold is applied.

    sensitive_map: SensitiveMap
        Class containing information about the minority and majority groups - used to match minority/majority columns.

    Notes
    -----

    Alternative names: *Preditive Equality*

    References
    ----------
    .. [1] Chouldechova, A., Benavides-Prado, D., Fialko, O., & Vaithianathan, R. (2018). A case study of algorithm-assisted
           decision making in child maltreatment hotline screening decisions. In FAccT (pp. 134-148).

    Examples
    --------
    >>> import torch
    >>> from torch_fairness.metrics import FalsePositiveRateBalance
    >>> from torch_fairness.data import SensitiveMap, SensitiveAttribute
    >>> sensitive = torch.tensor([[1., 0.], [1., 0.], [1., 0.], [0., 1.], [0., 1], [0., 1]])
    >>> pred = torch.tensor([0.7, 0.8, 0.2, 0.55, 0.7, 0.7])
    >>> labels = torch.tensor([1, 0, 0, 1, 0, 0])
    >>> sensitive_map = SensitiveMap(SensitiveAttribute(name='tests', majority=0, minority=[1]))
    >>> fair_loss = FalsePositiveRateBalance(threshold=0.5, sensitive_map=sensitive_map)
    >>> print(fair_loss(pred=pred, labels=labels, sensitive=sensitive))
    tensor([0.5000])
    """

    supports = ["classification"]

    def __init__(self, sensitive_map: SensitiveMap, threshold: Optional[float] = None):
        super().__init__(
            sensitive_map=sensitive_map, threshold=threshold, cm_metrics=["fpr"]
        )


class EqualOpportunity(ConfusionMatrixFairness):
    """Difference in false negative rates between majority and minority groups.

    .. math::
        P[\\hat{Y}=1|S=0,Y=1] = P[\\hat{Y}=1|S=1,Y=1]

    Parameters
    ----------
    threshold: float = None
        Threshold used for converting prediction into label. If None then no threshold is applied.

    sensitive_map: SensitiveMap
        Class containing information about the minority and majority groups - used to match minority/majority columns.

    Notes
    -----

    Depending upon whether a threshold is applied and the  activation function this can correspond to different
    measures:

    * If no threshold is applied and pred in {0,1} then this is generalized equal opportunity
    * If a threshold is applied, this is an indicator based approach [2, 3]
    * If a threshold and a smoothing of the prediction are used [4]

    Alternative names: *False negative rate balance*

    References
    ----------
    .. [1] Hardt, M., Price, E., & Srebro, N. (2016). Equality of opportunity in supervised learning. Advances in
        neural information processing systems, 29.
    .. [2] Lohaus, M., Perrot, M., & Von Luxburg, U. (2020, November). Too relaxed to be fair. In International
        Conference on Machine Learning (pp. 6360-6369). PMLR.
    .. [3] Donini, M., Oneto, L., Ben-David, S., Shawe-Taylor, J. S., & Pontil, M. (2018). Empirical risk minimization
        under fairness constraints. Advances in Neural Information Processing Systems, 31.
    .. [4] Wu, Y., Zhang, L., & Wu, X. (2019, May). On convexity and bounds of fairness-aware classification.
        In The World Wide Web Conference (pp. 3356-3362).

    Examples
    --------
    >>> import torch
    >>> from torch_fairness.metrics import EqualOpportunity
    >>> from torch_fairness.data import SensitiveMap, SensitiveAttribute
    >>> sensitive = torch.tensor([[1., 0.], [1., 0.], [1., 0.], [0., 1.], [0., 1], [0., 1]])
    >>> pred = torch.tensor([0.7, 0.8, 0.2, 0.55, 0.7, 0.7])
    >>> labels = torch.tensor([0, 1, 1, 0, 1, 1])
    >>> sensitive_map = SensitiveMap(SensitiveAttribute(name='tests', majority=0, minority=[1]))
    >>> fair_loss = EqualOpportunity(threshold=0.5, sensitive_map=sensitive_map)
    >>> print(fair_loss(pred=pred, labels=labels, sensitive=sensitive))
    tensor([0.5000])
    """

    supports = ["classification"]

    def __init__(self, sensitive_map: SensitiveMap, threshold: Optional[float] = None):
        super().__init__(
            sensitive_map=sensitive_map, threshold=threshold, cm_metrics=["fnr"]
        )


class EqualizedOdds(ConfusionMatrixFairness):
    """Average difference of true positive and false positive rates between minority and majority groups [1,2,3,4].

    .. math::
        P[\\hat{Y}=1|S=0,Y=1] = P[\\hat{Y}=1|S=1,Y=1], y \\in {0,1}

    Parameters
    ----------
    threshold: float = None
        Threshold used for converting prediction into label. If None then no threshold is applied.

    sensitive_map: SensitiveMap
        Class containing information about the minority and majority groups - used to match minority/majority columns.

    Notes
    -----

    Alternative names:

    * *Equalized odds* [1]
    * *Disparate mistreatment* [4]
    * *Conditional procedure accuracy equality*
    * *Sufficiency* [3]

    References
    ----------
    .. [1] Hardt, M., Price, E., & Srebro, N. (2016). Equality of opportunity in supervised learning. Advances in
        neural information processing systems, 29.
    .. [2] Donini, M., Oneto, L., Ben-David, S., Shawe-Taylor, J. S., & Pontil, M. (2018). Empirical risk minimization
        under fairness constraints. Advances in Neural Information Processing Systems, 31.
    .. [3] Solon Barocas, Moritz Hardt, and Arvind Naranayan. 2018. Fairness in MachineLearning. http://fairmlbook.org.
        (2018).
    .. [4] Zafar, M. B., Valera, I., Gomez Rodriguez, M., & Gummadi, K. P. (2017, April). Fairness beyond disparate
        treatment & disparate impact: Learning classification without disparate mistreatment. In Proceedings of the
        26th international conference on world wide web (pp. 1171-1180).

    Examples
    --------
    >>> import torch
    >>> from torch_fairness.metrics import EqualizedOdds
    >>> from torch_fairness.data import SensitiveMap, SensitiveAttribute
    >>> sensitive = torch.tensor([[1., 0.], [1., 0.], [1., 0.], [0., 1.], [0., 1], [0., 1]])
    >>> pred = torch.tensor([0.7, 0.8, 0.2, 0.55, 0.7, 0.7])
    >>> labels = torch.tensor([0, 1, 1, 0, 1, 1])
    >>> sensitive_map = SensitiveMap(SensitiveAttribute(name='tests', majority=0, minority=[1]))
    >>> fair_loss = EqualizedOdds(threshold=0.5, sensitive_map=sensitive_map)
    >>> print(fair_loss(pred=pred, labels=labels, sensitive=sensitive))
    tensor([0.2500])
    """

    supports = ["classification"]

    def __init__(self, sensitive_map: SensitiveMap, threshold: Optional[float] = None):
        super().__init__(
            sensitive_map=sensitive_map, threshold=threshold, cm_metrics=["fpr", "tpr"]
        )


class AccuracyEquality(ConfusionMatrixFairness):
    """Difference in overall accuracy between minority and majority groups [1].

    .. math::
        P[\\hat{Y}=Y|S=0] = P[\\hat{Y}=Y|S=1]

    Parameters
    ----------
    threshold: float = None
        Threshold used for converting prediction into label. If None then no threshold is applied.

    sensitive_map: SensitiveMap
        Class containing information about the minority and majority groups - used to match minority/majority columns.

    References
    ----------
    .. [1] Berk, R., Heidari, H., Jabbari, S., Kearns, M., & Roth, A. (2021). Fairness in criminal justice risk assessments:
        The state of the art. Sociological Methods & Research, 50(1), 3-44.

    Examples
    --------
    >>> import torch
    >>> from torch_fairness.metrics import AccuracyEquality
    >>> from torch_fairness.data import SensitiveMap, SensitiveAttribute
    >>> sensitive = torch.tensor([[1., 0.], [1., 0.], [1., 0.], [0., 1.], [0., 1], [0., 1]])
    >>> pred = torch.tensor([0.7, 0.8, 0.2, 0.55, 0.7, 0.7])
    >>> labels = torch.tensor([0, 1, 1, 0, 1, 1])
    >>> sensitive_map = SensitiveMap(SensitiveAttribute(name='tests', majority=0, minority=[1]))
    >>> fair_loss = AccuracyEquality(threshold=0.5, sensitive_map=sensitive_map)
    >>> print(fair_loss(pred=pred, labels=labels, sensitive=sensitive))
    tensor([0.3333])
    """

    supports = ["classification"]

    def __init__(self, sensitive_map: SensitiveMap, threshold: Optional[float] = None):
        super().__init__(
            sensitive_map=sensitive_map, threshold=threshold, cm_metrics=["accuracy"]
        )


class PredictiveParity(ConfusionMatrixFairness):
    """Difference in positive predictive values between majority and minority groups [1].

    .. math::
        P[\\hat{Y}=Y | \\hat{Y}=+, S=0] = P[\\hat{Y}=Y | \\hat{Y}=+, S=1]

    Parameters
    ----------
    threshold: float = None
        Threshold used for converting prediction into label. If None then no threshold is applied.

    sensitive_map: SensitiveMap
        Class containing information about the minority and majority groups - used to match minority/majority columns.

    Notes
    -----

    Alternative names: *Outcome tests*, *False discover rate balance*

    References
    ----------
    .. [1] Chouldechova, A. (2017). Fair prediction with disparate impact: A study of bias in recidivism prediction
        instruments. Big data, 5(2), 153-163.

    Examples
    --------
    >>> import torch
    >>> from torch_fairness.metrics import PredictiveParity
    >>> from torch_fairness.data import SensitiveMap, SensitiveAttribute
    >>> sensitive = torch.tensor([[1., 0.], [1., 0.], [1., 0.], [0., 1.], [0., 1], [0., 1]])
    >>> pred = torch.tensor([0.7, 0.8, 0.2, 0.55, 0.7, 0.7])
    >>> labels = torch.tensor([0, 1, 1, 0, 1, 1])
    >>> sensitive_map = SensitiveMap(SensitiveAttribute(name='tests', majority=0, minority=[1]))
    >>> fair_loss = PredictiveParity(threshold=0.5, sensitive_map=sensitive_map)
    >>> print(fair_loss(pred=pred, labels=labels, sensitive=sensitive))
    tensor([0.1667])
    """

    supports = ["classification"]

    def __init__(self, sensitive_map: SensitiveMap, threshold: Optional[float] = None):
        super().__init__(
            sensitive_map=sensitive_map, threshold=threshold, cm_metrics=["ppv"]
        )


class ConditionalUseAccuracyParity(ConfusionMatrixFairness):
    """Average difference of positive and negative predictive values between minority and majority groups [1].

    .. math::
        P[\\hat{Y}=Y | \\hat{Y}=+, S=0] = P[\\hat{Y}=Y | \\hat{Y}=+, S=1]

    .. math::
        P[\\hat{Y}=Y | \\hat{Y}=-, S=0] = P[\\hat{Y}=Y | \\hat{Y}=-, S=1]

    Parameters
    ----------
    threshold: float = None
        Threshold used for converting prediction into label. If None then no threshold is applied.

    sensitive_map: SensitiveMap
        Class containing information about the minority and majority groups - used to match minority/majority columns.

    References
    ----------
    .. [1] Chouldechova, A. (2017). Fair prediction with disparate impact: A study of bias in recidivism prediction
        instruments. Big data, 5(2), 153-163.

    Notes
    -----

    Alternative names: *Overall predictive parity*

    Examples
    --------
    >>> import torch
    >>> from torch_fairness.metrics import ConditionalUseAccuracyParity
    >>> from torch_fairness.data import SensitiveMap, SensitiveAttribute
    >>> sensitive = torch.tensor([[1., 0.], [1., 0.], [1., 0.], [1., 0.], [0., 1.], [0., 1], [0., 1], [0., 1]])
    >>> pred = torch.tensor([0.7, 0.8, 0.2, 0.8, 0.55, 0.7, 0.7, 0.2])
    >>> labels = torch.tensor([0, 1, 1, 0, 0, 1, 1, 0])
    >>> sensitive_map = SensitiveMap(SensitiveAttribute(name='tests', majority=0, minority=[1]))
    >>> fair_loss = ConditionalUseAccuracyParity(threshold=0.5, sensitive_map=sensitive_map)
    >>> print(fair_loss(pred=pred, labels=labels, sensitive=sensitive))
    tensor([0.6667])
    """

    supports = ["classification"]

    def __init__(self, sensitive_map: SensitiveMap, threshold: Optional[float] = None):
        super().__init__(
            sensitive_map=sensitive_map, threshold=threshold, cm_metrics=["ppv", "npv"]
        )


class BalancedPositive(BaseFairnessLoss):
    """Balanced predictions for positive labels [1].

    .. math::
        E[\\hat{Y} | Y=1, S=1] = E[\\hat{Y} | Y=1, S=0]

    Parameters
    ----------
    threshold: float = None
        Threshold used for converting prediction into label. If None then no threshold is applied.

    sensitive_map: SensitiveMap
        Class containing information about the minority and majority groups - used to match minority/majority columns.

    References
    ----------
    .. [1] Kleinberg, J., Mullainathan, S., & Raghavan, M. (2016). Inherent trade-offs in the fair determination of
        risk scores. arXiv preprint arXiv:1609.05807.

    Examples
    --------
    >>> import torch
    >>> from torch_fairness.metrics import BalancedPositive
    >>> from torch_fairness.data import SensitiveMap, SensitiveAttribute
    >>> sensitive = torch.tensor([[1, 0], [1, 0], [0, 1], [0, 1]])
    >>> pred = torch.tensor([0.2, 0.8, 1., 0.8])
    >>> labels = torch.tensor([1., 0., 1., 0.])
    >>> sensitive_map = SensitiveMap(SensitiveAttribute(name='tests', majority=0, minority=[1]))
    >>> fair_loss = BalancedPositive(sensitive_map=sensitive_map)
    >>> print(fair_loss(pred=pred, labels=labels, sensitive=sensitive))
    tensor([0.8000])
    """

    supports = ["classification"]

    def __init__(self, sensitive_map: SensitiveMap, threshold: Optional[float] = None):
        super().__init__(sensitive_map=sensitive_map)
        self.threshold_layer = ThresholdOperator(threshold=threshold)

    def forward(
        self, pred: torch.tensor, labels: torch.tensor, sensitive: torch.tensor
    ) -> torch.tensor:
        """
        Parameters
        ----------
        pred: torch.tensor
            Predictions.

        sensitive: torch.tensor
            Dummy-coded minority/majority sensitive groups.

        labels: torch.tensor
            Criterion labels.

        Returns
        -------
        Group fairness loss: torch.tensor
        """
        pred = _validate_shape(pred, squeeze=True)
        labels = _validate_shape(labels, squeeze=True)
        result = []

        pred = pred[labels.squeeze() == 1]
        sensitive = sensitive[labels == 1, :]
        pred = self.threshold_layer(pred)
        for maj_idx, min_groups in self.sensitive_map.majority_minority_pairs:
            maj_sensitive = sensitive[:, maj_idx]
            majority = get_member(maj_sensitive, pred)
            for min_idx in min_groups:
                min_sensitive = sensitive[:, min_idx]
                minority = get_member(min_sensitive, pred)
                disparity = torch.abs(minority.mean() - majority.mean())
                result.append(disparity)
        result = torch.stack(result)
        return result


class BalancedNegative(BaseFairnessLoss):
    """Balanced predictions for negative labels [1].

    .. math::
        E[\\hat{Y} | Y=0,S=1] = E[\\hat{Y} | Y=0, S=0]

    Parameters
    ----------
    threshold: float = None
        Threshold used for converting prediction into label. If None then no threshold is applied.

    sensitive_map: SensitiveMap
        Class containing information about the minority and majority groups - used to match minority/majority columns.

    References
    ----------
    .. [1] Kleinberg, J., Mullainathan, S., & Raghavan, M. (2016). Inherent trade-offs in the fair determination of
        risk scores. arXiv preprint arXiv:1609.05807.

    Examples
    --------
    >>> import torch
    >>> from torch_fairness.metrics import BalancedNegative
    >>> from torch_fairness.data import SensitiveMap, SensitiveAttribute
    >>> sensitive = torch.tensor([[1, 0], [1, 0], [0, 1], [0, 1]])
    >>> pred = torch.tensor([0.2, 0.8, 1., 0.8])
    >>> labels = torch.tensor([1., 0., 1., 0.])
    >>> sensitive_map = SensitiveMap(SensitiveAttribute(name='tests', majority=0, minority=[1]))
    >>> fair_loss = BalancedNegative(sensitive_map=sensitive_map)
    >>> print(fair_loss(pred=pred, labels=labels, sensitive=sensitive))
    tensor([0.])
    """

    supports = ["classification"]

    def __init__(self, sensitive_map: SensitiveMap, threshold: Optional[float] = None):
        super().__init__(sensitive_map=sensitive_map)
        self.threshold_layer = ThresholdOperator(threshold=threshold)

    def forward(
        self, pred: torch.tensor, labels: torch.tensor, sensitive: torch.tensor
    ) -> torch.tensor:
        """
        Parameters
        ----------
        pred: torch.tensor
            Predictions.

        sensitive: torch.tensor
            Dummy-coded minority/majority sensitive groups.

        labels: torch.tensor
            Criterion labels.

        Returns
        -------
        Group fairness loss: torch.tensor
        """
        pred = _validate_shape(pred, squeeze=True)
        labels = _validate_shape(labels, squeeze=True)
        result = []

        pred = pred[labels.squeeze() == 0]
        sensitive = sensitive[labels == 0, :]
        pred = self.threshold_layer(pred)
        for maj_idx, min_groups in self.sensitive_map.majority_minority_pairs:
            maj_sensitive = sensitive[:, maj_idx]
            majority = get_member(maj_sensitive, pred)
            for min_idx in min_groups:
                min_sensitive = sensitive[:, min_idx]
                minority = get_member(min_sensitive, pred)
                disparity = torch.abs(minority.mean() - majority.mean())
                result.append(disparity)
        result = torch.stack(result)
        return result


class DemographicParity(BaseFairnessLoss):
    """
    The absolute difference in proportion of group members with :math:`\\hat{Y}` =1 between minority and majority groups
    [1, 2].

    Demographic parity is also commonly calculated using ratios of the selection rate between the minority and majority,
    such as in the four-fifths ratio. However, the ratio is asymmetric between the majority/minority groups, which is
    undesirable when the goal is to treat groups equally. Using the difference in selection proportions circumvents this
    problem.

    .. math::
        P[\\hat{Y} | S=1] = P[\\hat{Y}|S=0]

    Parameters
    ----------
    threshold: float = None
        Threshold used for converting prediction into label. If None then no threshold is applied.

    sensitive_map: SensitiveMap
        Class containing information about the minority and majority groups - used to match minority/majority columns.

    Notes
    -----

    Alternative names: *statistical parity*

    References
    ----------
    .. [1] Wu, Y., Zhang, L., & Wu, X. (2019). On convexity and bounds of fairness-aware classification. In The World
        Wide Web Conference (pp. 3356-3362).
    .. [2] Dwork, C., Hardt, M., Pitassi, T., Reingold, O., & Zemel, R. (2012). Fairness through awareness. In Proceedings
        of the 3rd innovations in theoretical computer science conference (pp. 214-226).

    Examples
    --------
    >>> import torch
    >>> from torch_fairness.metrics import DemographicParity
    >>> from torch_fairness.data import SensitiveMap, SensitiveAttribute
    >>> sensitive = torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 1]])
    >>> pred = torch.tensor([0.7, 0.6, 0.6, 0.2])
    >>> sensitive_map = SensitiveMap(SensitiveAttribute(name='tests', majority=0, minority=[1]))
    >>> fair_loss = DemographicParity(threshold=0.5, sensitive_map=sensitive_map)
    >>> print(fair_loss(pred=pred, sensitive=sensitive))
    tensor([0.5000])
    """

    supports = ["classification"]

    def __init__(self, threshold: Optional[float] = None, **kwargs):
        super().__init__(**kwargs)
        self.threshold_layer = ThresholdOperator(threshold=threshold)

    @staticmethod
    def selection_rate(pred: torch.tensor) -> torch.tensor:
        return pred.sum() / pred.shape[0]

    def forward(self, pred: torch.tensor, sensitive: torch.tensor) -> torch.tensor:
        """
        Parameters
        ----------
        pred: torch.tensor
            Predictions.

        sensitive: torch.tensor
            Dummy-coded minority/majority sensitive groups.

        Returns
        -------
        Group fairness loss: torch.tensor
        """
        pred = _validate_shape(pred, squeeze=True)
        result = []
        pred = self.threshold_layer(pred)
        for maj_idx, min_groups in self.sensitive_map.majority_minority_pairs:
            maj_sensitive = sensitive[:, maj_idx]
            majority = self.selection_rate(get_member(maj_sensitive, pred))
            for min_idx in min_groups:
                min_sensitive = sensitive[:, min_idx]
                minority = self.selection_rate(get_member(min_sensitive, pred))
                disparity = torch.abs(minority - majority)
                result.append(disparity)
        result = torch.stack(result)
        return result


class SmoothedEmpiricalDifferentialFairness(BaseFairnessLoss):
    """Smoothed version of demographic parity that helps with stability when there are 0 counts [1].

    Parameters
    ----------
    threshold: float = None
        Threshold used for converting prediction into label. If None then no threshold is applied.

    sensitive_map: SensitiveMap
        Class containing information about the minority and majority groups - used to match minority/majority columns.

    Notes
    -----

    The proposed method in [1] involves (a) calculating the smoothed empirical fairness measures across each group then
    (b) taking the maximum across the groups. This allows the authors to support intersectional groups where the number
    of groups rises rapidly. This class only performs step (a) as the minimax approach may not match every problem.

    References
    ----------
    .. [1] Foulds, J. R., Islam, R., Keya, K. N., & Pan, S. (2020). An intersectional definition of fairness.
        In 2020 IEEE 36th International Conference on Data Engineering (ICDE) (pp. 1918-1921). IEEE.

    Examples
    --------
    >>> import torch
    >>> from torch_fairness.metrics import SmoothedEmpiricalDifferentialFairness
    >>> from torch_fairness.data import SensitiveMap, SensitiveAttribute
    >>> sensitive = torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 1]])
    >>> pred = torch.tensor([1, 1, 0, 1])
    >>> sensitive_map = SensitiveMap(SensitiveAttribute(name='tests', majority=0, minority=[1]))
    >>> fair_loss = SmoothedEmpiricalDifferentialFairness(sensitive_map=sensitive_map)
    >>> print(fair_loss(pred=pred, sensitive=sensitive))
    tensor([1.0986])
    """

    supports = ["classification"]
    # Parameters used in original differential fairness paper
    num_classes: int = 2  # The number of outcomes available (selected / not selected)
    dirichlet_alpha: float = 0.5
    concentration_parameter: float = num_classes * dirichlet_alpha

    def __init__(self, threshold: Optional[float] = None, **kwargs):
        super().__init__(**kwargs)
        self.threshold_layer = ThresholdOperator(threshold=threshold)

    def _smooth_hire_prop(self, n_hire: torch.tensor, n: torch.tensor) -> torch.tensor:
        """
        Differential fairness converts the raw counts into a smoothed proportion of individuals hired based on the
        Dirichlet distribution. The parameters help with stability when counts are small.

        The parameters here are based on what was used in the paper.
        """
        return (n_hire + self.dirichlet_alpha) / (n + self.concentration_parameter)

    def forward(self, pred: torch.tensor, sensitive: torch.tensor) -> torch.tensor:
        """
        Parameters
        ----------
        pred: torch.tensor
            Predictions.

        sensitive: torch.tensor
            Dummy-coded minority/majority sensitive groups.

        Returns
        -------
        Group fairness loss: torch.tensor
        """
        pred = _validate_shape(pred, squeeze=True)
        result = []
        pred = self.threshold_layer(pred)
        for maj_idx, min_groups in self.sensitive_map.majority_minority_pairs:
            maj_sensitive = sensitive[:, maj_idx]
            n_hire_majority = get_member(maj_sensitive, pred).sum()
            n_majority = torch.sum(get_member(maj_sensitive))
            prop_hire_majority = self._smooth_hire_prop(n_hire_majority, n_majority)
            for min_idx in min_groups:
                min_sensitive = sensitive[:, min_idx]
                n_hire_minority = get_member(min_sensitive, pred).sum()
                n_minority = torch.sum(get_member(min_sensitive))
                prop_hire_minority = self._smooth_hire_prop(n_hire_minority, n_minority)
                # Differential fairness loss
                epsilon = torch.log(prop_hire_majority) - torch.log(prop_hire_minority)
                epsilon2 = torch.log(1 - prop_hire_majority) - torch.log(
                    1 - prop_hire_minority
                )
                differential_fairness = torch.max(
                    torch.abs(torch.stack([epsilon, epsilon2]))
                )
                result.append(differential_fairness)
        result = torch.stack(result)
        return result


class WeightedSumofLogs(BaseFairnessLoss):
    """Convex approximation of demographic parity using weights calculated based on historical bias.

    Parameters
    ----------
    labels: torch.tensor
        Existing labels are needed to calculate historical bias.

    sensitive: torch.tensor
        Sensitive paired with existing labels used to calculate historical bias.

    sensitive_map: SensitiveMap
        Class containing information about the minority and majority groups - used to match minority/majority columns.

    reduce: bool = True
        Whether to reduce-instance level loss across sensitive groups. If True, then each individual has a single loss
        value.

    Notes
    -----

    This approach uses an instance-level loss function, unlike most fairness measures which are list-wise.

    References
    ----------
    .. [1] Goel, N., Yaghini, M., & Faltings, B. (2018). Non-discriminatory machine learning through convex
        fairness criteria. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 32, No. 1).

    Examples
    --------
    >>> import torch
    >>> from torch_fairness.metrics import WeightedSumofLogs
    >>> from torch_fairness.data import SensitiveMap, SensitiveAttribute
    >>> sensitive = torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 1]])
    >>> pred = torch.tensor([0.7, 0.6, 0.6, 0.2])
    >>> labels = torch.tensor([1, 0, 0, 1])
    >>> sensitive_map = SensitiveMap(SensitiveAttribute(name='tests', majority=0, minority=[1]))
    >>> fair_loss = WeightedSumofLogs(labels=labels, sensitive=sensitive, sensitive_map=sensitive_map)
    >>> print(fair_loss(pred=pred, sensitive=sensitive))
    tensor([[0.0446],
            [0.0639],
            [0.0639],
            [0.2012]])
    """

    supports = ["classification"]

    def __init__(
        self,
        labels: torch.tensor,
        sensitive: torch.tensor,
        sensitive_map: SensitiveMap,
        reduce: bool = True,
    ):
        super().__init__(sensitive_map=sensitive_map)
        self.historical_bias = self._calculate_historical_bias(labels, sensitive)
        self.reduce = reduce

    def _calculate_historical_bias(
        self, labels: torch.tensor, sensitive: torch.tensor
    ) -> torch.tensor:
        labels = _validate_shape(labels)
        member_idx = get_member(sensitive)
        sample_size = member_idx.sum(0)
        negative_label = (member_idx & (labels.unsqueeze(1) == 0)).sum(0)
        historical_bias = negative_label / sample_size**2
        return historical_bias

    def forward(self, pred: torch.tensor, sensitive: torch.tensor) -> torch.tensor:
        """
        Parameters
        ----------
        pred: torch.tensor
            Predictions.

        sensitive: torch.tensor
            Dummy-coded minority/majority sensitive groups.

        Returns
        -------
        Group fairness loss: torch.tensor
        """
        pred = _validate_shape(pred, squeeze=False)
        bias = sensitive * self.historical_bias
        # Some sensitive groups may be missing - set these nan to 0 so that they don't cause issues
        bias = torch.nan_to_num(bias, 0.0)
        if self.reduce:
            bias = torch.mean(bias, 1).unsqueeze(1)
        loss = -bias * torch.log(pred)
        return loss

    def calculate(
        self, pred: torch.tensor, historical_bias: torch.tensor
    ) -> torch.tensor:
        loss = -pred * historical_bias
        loss = self.agg_func(loss)
        return loss


class MeanDifferences(BaseFairnessLoss):
    """Difference in mean between minority and majority groups.

    .. math::
        E[\\hat{Y}|S=0] = E[\\hat{Y}|S=1]

    Parameters
    ----------
    sensitive_map: SensitiveMap
        Class containing information about the minority and majority groups - used to match minority/majority columns.

    Notes
    -----

    Depending upon the input tensor this can produce different measurements.

    * If predictions: this is mean differences [1], an indepdence measurement
    * If loss: this is bounded group loss [2], a seperation measurement
    * If residuals: this is balanced residuals [3], a seperation measurement

    References
    ----------
    .. [1] Calders, T., Karim, A., Kamiran, F., Ali, W., & Zhang, X. (2013). Controlling attribute effect in
           linear regression. In 2013 IEEE 13th international conference on data mining (pp. 71-80). IEEE.
    .. [2] Agarwal, A., Dudík, M., & Wu, Z. S. (2019, May). Fair regression: Quantitative definitions and reduction-based
           algorithms. In International Conference on Machine Learning (pp. 120-129). PMLR.
    .. [3] Calders, T., Karim, A., Kamiran, F., Ali, W., & Zhang, X. (2013, December). Controlling attribute effect in
           linear regression. In 2013 IEEE 13th international conference on data mining (pp. 71-80). IEEE.

    Examples
    --------
    >>> import torch
    >>> from torch_fairness.metrics import MeanDifferences
    >>> from torch_fairness.data import SensitiveMap, SensitiveAttribute
    >>> x = torch.tensor([0., 2., -1., 1.])
    >>> sensitive = torch.tensor([[1, 0], [1, 0], [0, 1], [0, 1]])
    >>> sensitive_map = SensitiveMap(SensitiveAttribute(name='tests', minority=[0], majority=1))
    >>> fair_loss = MeanDifferences(sensitive_map=sensitive_map)
    >>> print(fair_loss(x, sensitive=sensitive))
    tensor([1.])
    """

    supports = ["regression"]

    def calculate(self, majority: torch.tensor, minority: torch.tensor) -> torch.tensor:
        return torch.abs(majority.mean() - minority.mean())

    def forward(self, pred: torch.tensor, sensitive: torch.tensor) -> torch.tensor:
        """
        Parameters
        ----------
        pred: torch.tensor
            Predictions.

        sensitive: torch.tensor
            Dummy-coded minority/majority sensitive groups.

        Returns
        -------
        Group fairness loss: torch.tensor
        """
        pred = _validate_shape(pred, squeeze=True)

        result = []
        for maj_idx, min_groups in self.sensitive_map.majority_minority_pairs:
            maj_sensitive = sensitive[:, maj_idx]
            majority = get_member(maj_sensitive, pred)
            for min_idx in min_groups:
                sensitive_minority = sensitive[:, min_idx]
                minority = get_member(sensitive_minority, pred)
                group_loss = self.calculate(majority=majority, minority=minority)
                result.append(group_loss)
        result = torch.stack(result)
        return result


class CrossPairDistance(BaseFairnessLoss):
    """Calculates the pairwise comparisons between minority and majority samples using labels and predictions [1].

    Parameters
    ----------
    sensitive_map: SensitiveMap
        Class containing information about the minority and majority groups - used to match minority/majority columns.

    Notes
    -----

    References
    ----------
    .. [1] Berk, R., Heidari, H., Jabbari, S., Joseph, M., Kearns, M., Morgenstern, J., ... & Roth, A. (2017). A convex
        framework for fair regression. arXiv preprint arXiv:1706.02409.

    Examples
    --------
    >>> import torch
    >>> from torch_fairness.metrics import CrossPairDistance
    >>> from torch_fairness.data import SensitiveMap, SensitiveAttribute
    >>> sensitive = torch.tensor([[1., 0.], [1., 0.], [1., 0.], [0., 1.], [0., 1.], [0., 1.]])
    >>> labels = torch.tensor([-1, 1.2, 1.5, -0.2, 0.6, 0])
    >>> pred = torch.tensor([-1.5, 1.0, 1.0, 0, 0.2, 0.5])
    >>> sensitive_map = SensitiveMap(SensitiveAttribute(name='tests', majority=0, minority=[1]))
    >>> fair_loss = CrossPairDistance(sensitive_map=sensitive_map)
    >>> print(fair_loss(pred=pred, labels=labels, sensitive=sensitive))
    tensor([0.0017])
    """

    supports = ["classification", "regression"]

    def __init__(self, is_regression: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_regression = is_regression

    @staticmethod
    def pairwise_difference(x: torch.tensor, y: torch.tensor) -> torch.tensor:
        return x[:, None] - y[None, :]

    def pairwise_label_distance(self, x: torch.tensor, y: torch.tensor) -> torch.tensor:
        if self.is_regression:
            dist = self.pairwise_difference(x, y)
            dist = torch.exp(-torch.pow(dist, 2))
        else:
            dist = 1.0 * (x[:, None] == y[None, :])
        return dist

    def forward(
        self, pred: torch.tensor, labels: torch.tensor, sensitive: torch.tensor
    ) -> torch.tensor:
        """
        Parameters
        ----------
        pred: torch.tensor
            Predictions.

        sensitive: torch.tensor
            Dummy-coded minority/majority sensitive groups.

        labels: torch.tensor
            Criterion labels.

        Returns
        -------
        Group fairness loss: torch.tensor
        """
        pred = _validate_shape(pred, squeeze=True)
        labels = _validate_shape(labels, squeeze=True)
        if pred.shape != labels.shape:
            raise ValueError(
                f"Pred ({pred.shape}) and labels ({labels.shape}) must be the same size."
            )
        result = []
        for maj_idx, min_groups in self.sensitive_map.majority_minority_pairs:
            maj_sensitive = sensitive[:, maj_idx]
            maj_pred = get_member(maj_sensitive, pred)
            maj_labels = get_member(maj_sensitive, labels)
            for min_idx in min_groups:
                min_sensitive = sensitive[:, min_idx]
                min_pred = get_member(min_sensitive, pred)
                min_labels = get_member(min_sensitive, labels)
                prediction_difference = self.pairwise_difference(maj_pred, min_pred)
                label_distance = self.pairwise_label_distance(maj_labels, min_labels)
                loss = torch.mean(prediction_difference * label_distance)
                loss = torch.pow(loss, 2)
                result.append(loss)
        result = torch.stack(result)
        return result


class AbsoluteCorrelation(BaseFairnessLoss):
    """Absolute correlation between predictions and sensitive group [1].

    Parameters
    ----------
    sensitive_map: SensitiveMap
        Class containing information about the minority and majority groups - used to match minority/majority columns.

    References
    ----------
    .. [1] Beutel, A., Chen, J., Doshi, T., Qian, H., Woodruff, A., Luu, C., ... & Chi, E. H. (2019). Putting fairness
        principles into practice: Challenges, metrics, and improvements. In Proceedings of the 2019 AAAI/ACM
        Conference on AI, Ethics, and Society (pp. 453-459).

    Examples
    --------
    >>> import torch
    >>> from torch_fairness.metrics import AbsoluteCorrelation
    >>> from torch_fairness.data import SensitiveMap, SensitiveAttribute
    >>> pred = torch.tensor([0.7, 1., 0.2, 0.55, 0.7, 0.7])
    >>> labels = torch.tensor([0, 1, 1, 0, 1, 1])
    >>> sensitive = torch.tensor([[1., 0.], [1., 0.], [1., 0.], [0., 1.], [0., 1], [0., 1]])
    >>> sensitive_map = SensitiveMap(SensitiveAttribute(name='tests', majority=0, minority=[1]))
    >>> fair_loss = AbsoluteCorrelation(sensitive_map=sensitive_map)
    >>> print(fair_loss(pred=pred, sensitive=sensitive))
    tensor([0.0349])
    """

    epsilon = 1e-7

    def forward(self, pred: torch.tensor, sensitive: torch.tensor) -> torch.tensor:
        """
        Parameters
        ----------
        pred: torch.tensor
            Tensor of predictions.

        sensitive: torch.tensor
            Tensor of dummy-coded minority/majority sensitive groups.

        Returns
        -------
        Group fairness loss: torch.tensor
        """
        pred = _validate_shape(pred, squeeze=True)
        result = []
        sensitive_observed = ~torch.isnan(sensitive)
        for maj_idx, min_groups in self.sensitive_map.majority_minority_pairs:
            for min_idx in min_groups:
                sensitive_observed_i = sensitive_observed[:, min_idx]
                sensitive_minority = sensitive[:, min_idx]
                absolute_correlation = self.calculate(
                    pred[sensitive_observed_i].squeeze(),
                    sensitive_minority[sensitive_observed_i],
                )
                result.append(absolute_correlation)
        result = torch.stack(result)
        return result

    def calculate(self, x: torch.tensor, y: torch.tensor) -> torch.tensor:
        x_dist = x - torch.mean(x)
        y_dist = y - torch.mean(y)
        x_var = torch.sqrt(torch.sum(x_dist**2) + self.epsilon)
        y_var = torch.sqrt(torch.sum(y_dist**2) + self.epsilon)
        cost = torch.sum(x_dist * y_dist) / (x_var * y_var)
        cost = torch.abs(cost)
        return cost


# TODO - Create example
class MMDFairness(BaseFairnessLoss):
    """Reduce the MMD between majority and minority groups [1].

    Currently only supports a Gaussian kernel.

    Parameters
    ----------

    sensitive_map: SensitiveMap
        Class containing information about the minority and majority groups - used to match minority/majority columns.
    bandwidth: float = 1.
        Width of gaussian kernel used.

    References
    ----------
    .. [1] Louizos, C., Swersky, K., Li, Y., Welling, M., & Zemel, R. (2016). The variational fair autoencoder. ICLR.

    Examples
    --------
    >>> import torch
    >>> from torch_fairness.metrics import MMDFairness
    >>> from torch_fairness.data import SensitiveMap, SensitiveAttribute
    >>> sensitive_map = SensitiveMap(SensitiveAttribute(name='tests', majority=0, minority=[1]))
    >>> sensitive = torch.tensor([[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1]])
    >>> pred = torch.tensor([0., 1., -1., 0., 2., -2.])
    >>> fair_loss = MMDFairness(sensitive_map=sensitive_map)
    >>> print(fair_loss(x=pred, sensitive=sensitive))
    tensor([0.2850])
    """

    supports = ["classification", "regression"]
    kernel = "gaussian"

    def __init__(self, sensitive_map, biased: bool = True, bandwidth: float = 1.0):
        self.biased = biased
        self.bandwidth = bandwidth
        super().__init__(sensitive_map=sensitive_map)

    def forward(self, x: torch.tensor, sensitive: torch.tensor) -> torch.tensor:
        x = _validate_shape(x, squeeze=False)
        result = []
        for maj_idx, min_groups in self.sensitive_map.majority_minority_pairs:
            maj_sensitive = sensitive[:, maj_idx]
            majority = get_member(maj_sensitive, x)
            for min_idx in min_groups:
                sensitive_minority = sensitive[:, min_idx]
                minority = get_member(sensitive_minority, x)
                group_loss = self.calculate(majority=majority, minority=minority)
                result.append(group_loss)
        result = torch.stack(result)
        return result

    def calculate(self, minority: torch.tensor, majority: torch.tensor) -> torch.tensor:
        n_majority = majority.shape[0]
        n_minority = minority.shape[0]
        # Distances
        majority_distances = F.pdist(majority, 2.0)
        minority_distances = F.pdist(minority, 2.0)
        cross_distances = torch.cdist(majority, minority, p=2.0)
        # Kernel over distances
        kernel_majority = self.gaussian_kernel(majority_distances)
        kernel_minority = self.gaussian_kernel(minority_distances)
        kernel_marginal = self.gaussian_kernel(cross_distances)
        # Aggregate
        if self.biased:
            kernel_majority = (
                kernel_majority.sum() * 2
                + n_majority * self.gaussian_kernel(torch.tensor(0))
            )
            kernel_minority = (
                kernel_minority.sum() * 2
                + n_minority * self.gaussian_kernel(torch.tensor(0))
            )
            kernel_marginal = kernel_marginal.sum()
            kernel_majority *= 1 / (n_majority * n_majority)
            kernel_minority *= 1 / (n_minority * n_minority)
            kernel_marginal *= 1 / (n_majority * n_minority)
        else:
            # Since pdist calculates all unique comparisons, we multiply times 2 to get sum along pairwise distance matrix
            # with diagonal removed. This is the unbiased estimator, however, if minority=majority is does not equal 0.
            kernel_majority = kernel_majority.sum() * 2
            kernel_minority = kernel_minority.sum() * 2
            kernel_marginal = kernel_marginal.sum()
            kernel_majority *= 1 / (n_majority * (n_majority - 1))
            kernel_minority *= 1 / (n_minority * (n_minority - 1))
            kernel_marginal *= 1 / (n_majority * n_minority)
        mmd = kernel_majority + kernel_minority - 2 * kernel_marginal
        return mmd

    def gaussian_kernel(self, pairwise_distance: torch.tensor) -> torch.tensor:
        """
        With a Gaussian kernel bandwidth is equivalent to 1/(Sigma**2*2)
        """
        return torch.exp(-self.bandwidth * torch.pow(pairwise_distance, 2))
