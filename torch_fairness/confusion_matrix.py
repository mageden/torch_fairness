
from typing import TypedDict
from typing import Optional
from typing import Literal

import torch


def _validate_input(labels, pred):
    if labels.shape != pred.shape:
        raise ValueError("Labels and pred must have same shape.")
    if labels.ndim != 1 and labels.ndim != 2:
        raise ValueError("Input must be a 1D or 2D array.")


def true_positive(labels: torch.tensor, pred: torch.tensor) -> torch.tensor:
    """Takes the sum of predictions when there is a positive label (assumed 1). If pred is binary this is the number of
    true positives, if predictions is continuous then it is the generalized true positives.

    Args:
        labels (torch.tensor): 1D tensor of labels - assumed to be binary (e.g., 0, 1, nan)
        pred (torch.tensor): 1D tensor of predictions

    Returns:
        torch.tensor: The return value - 1D tensor.
    """
    _validate_input(labels, pred)
    positive_labels = labels == 1
    positive_scores = pred*positive_labels
    tp = positive_scores.sum(0)
    return tp


def false_positive(labels: torch.tensor, pred: torch.tensor) -> torch.tensor:
    """Takes the sum of predictions when there is a negative label (assumed 0). If pred is binary this is the number of
    false positives, if predictions is continuous then it is the generalized false positives.

    Args:
        labels (torch.tensor): 1D tensor of labels - assumed to be binary (e.g., 0, 1, nan)
        pred (torch.tensor): 1D tensor of predictions

    Returns:
        torch.tensor: The return value - 1D tensor.
    """
    _validate_input(labels, pred)
    negative_labels = labels == 0
    positive_scores = pred*negative_labels
    fp = positive_scores.sum(0)
    return fp


def true_negative(labels: torch.tensor, pred: torch.tensor) -> torch.tensor:
    """Takes the sum of inverted predictions (1-pred) when there is a negative label (assumed 0). If pred is binary
    this is the number of true negatives, if pred is continuous then it is the generalized true negatives.

    Args:
        labels (torch.tensor): 1D tensor of labels - assumed to be binary (e.g., 0, 1, nan)
        pred (torch.tensor): 1D tensor of predictions

    Returns:
        torch.tensor: The return value - 1D tensor.
    """
    _validate_input(labels, pred)
    negative_labels = labels == 0
    negative_scores = (1-pred)*negative_labels
    tn = negative_scores.sum(0)
    return tn


def false_negative(labels: torch.tensor, pred: torch.tensor) -> torch.tensor:
    """Takes the sum of inverted predictions (1-pred) when there is a positive label (assumed 1). If pred is binary
    this is the number of true negatives, if pred is continuous then it is the generalized true negatives.

    Args:
        labels (torch.tensor): 1D tensor of labels - assumed to be binary (e.g., 0, 1, nan)
        pred (torch.tensor): 1D tensor of predictions

    Returns:
        torch.tensor: The return value - 1D tensor.
    """
    _validate_input(labels, pred)
    positive_labels = labels == 1
    negative_scores = (1-pred)*positive_labels
    fn = negative_scores.sum(0)
    return fn


def false_positive_rate(labels: torch.tensor, pred: torch.tensor) -> torch.tensor:
    number_of_negatives = torch.sum(1 - labels)
    false_positives = false_positive(labels, pred)
    return false_positives / number_of_negatives


def false_negative_rate(labels: torch.tensor, pred: torch.tensor) -> torch.tensor:
    number_of_positives = torch.sum(labels)
    fn = false_negative(labels, pred)
    return fn / number_of_positives


def true_positive_rate(labels: torch.tensor, pred: torch.tensor) -> torch.tensor:
    number_of_positives = torch.sum(labels)
    tp = true_positive(labels, pred)
    return tp / number_of_positives


def true_negative_rate(labels: torch.tensor, pred: torch.tensor) -> torch.tensor:
    number_of_negatives = torch.sum(1 - labels)
    tn = true_negative(labels, pred)
    return tn / number_of_negatives


class ConfusionMetrics(TypedDict):
    tpr: Optional[torch.tensor]
    fpr: Optional[torch.tensor]
    tnr: Optional[torch.tensor]
    fnr: Optional[torch.tensor]
    ppv: Optional[torch.tensor]
    npv: Optional[torch.tensor]
    accuracy: Optional[torch.tensor]


class ConfusionMatrix:
    """
    Convenience wrapper for when multiple metrics are being calculated to prevent recalculating or calculated
    unneeded metrics. If only using a single confusion metric (e.g., true_positive_rate) then just call the function.


    Parameters
    ----------
    metrics : List[Literal['tpr', 'fpr', 'tnr', 'fnr', 'ppv', 'npv', 'accuracy']]

    Examples
    --------
    >>> import torch
    >>> from torch_fairness.confusion_matrix import ConfusionMatrix

    >>> labels = torch.tensor([[1, 1, 0, 0]])
    >>> pred = torch.tensor([[1, 0, 1, 0]])
    >>> cm = ConfusionMatrix(metrics=['accuracy'])
    >>> results = cm.calculate(labels, pred)
    >>> results
    tensor([0.5])
    """
    def __init__(self, metrics: list[Literal['tpr', 'fpr', 'tnr', 'fnr', 'ppv', 'npv', 'accuracy']]):
        if metrics is None or len(metrics) == 0:
            raise ValueError("At least one metric must be specified")
        self.metrics = set(metrics)

    def calculate(self, labels: torch.tensor, pred: torch.tensor) -> ConfusionMetrics:
        # Calculate required components
        if not self.metrics.isdisjoint({'tpr', 'fnr', 'ppv', 'accuracy'}):
            tp = true_positive(labels, pred)
        if not self.metrics.isdisjoint({'fpr', 'tnr', 'ppv', 'accuracy'}):
            fp = false_positive(labels, pred)
        if not self.metrics.isdisjoint({'fpr', 'tnr', 'npv', 'accuracy'}):
            tn = true_negative(labels, pred)
        if not self.metrics.isdisjoint({'tpr', 'fnr', 'npv', 'accuracy'}):
            fn = false_negative(labels, pred)
        out = {}
        if 'tpr' in self.metrics:
            out['tpr'] = tp/(tp+fn)
        if 'fpr' in self.metrics:
            out['fpr'] = fp/(fp+tn)
        if 'tnr' in self.metrics:
            out['tnr'] = tn/(tn+fp)
        if 'fnr' in self.metrics:
            out['fnr'] = fn/(fn+tp)
        if 'ppv' in self.metrics:
            out['ppv'] = tp / (tp+fp)
        if 'npv' in self.metrics:
            out['npv'] = tn / (tn+fn)
        if 'accuracy' in self.metrics:
            out['accuracy'] = (tp+tn) / (tp+tn+fp+fn)
        return out
