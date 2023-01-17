import warnings
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from scipy.spatial import KDTree
from scipy.stats import mode

from torch_fairness.data import get_member
from torch_fairness.util import _validate_dummy_coded_data, set_random_state


def imbalance_ratio(
    sample_sizes: Union[np.ndarray, pd.DataFrame]
) -> Union[np.ndarray, pd.DataFrame]:
    """Imbalance Ratio per Lael (IRLbl) is the ratio between the sample sizes of each group and the largest group. The largest group
    always has a value of 1. and every other group has a value at least as large.

    .. math::
        IRLbl(S_{i}) = \\frac{\\underset{i \\in K}{\\text{max}}N_{i}}{N_{i}}


    Parameters
    ----------
    sample_sizes: np.ndarray
        Sample sizes across each group.

    Returns
    -------
    np.ndarray
        The information ratio across all groups.


    References
    ----------
    .. [1] Charte, F., Rivera, A. J., del Jesus, M. J., & Herrera, F. (2015). Addressing imbalance in multilabel
        classification: Measures and random resampling algorithms. Neurocomputing, 163, 3-16.

    Examples
    --------
    >>> from torch_fairness.resampling import imbalance_ratio
    >>> import numpy as np
    >>> sample_sizes = np.array([50, 200, 330])
    >>> print(imbalance_ratio(sample_sizes))
    [6.6  1.65 1.  ]

    """
    if len(sample_sizes.shape) != 1:
        raise ValueError(
            f"Sample size should be a 1D array - found a {len(sample_sizes.shape)}D instead."
        )
    if np.isnan(sample_sizes).mean() == 1.0:
        raise ValueError("Sample size is all nan - cannot compute information ratio.")
    ir = sample_sizes.max() / sample_sizes
    return ir


def geometric_mean(x: np.ndarray, axis=-1):
    # Take log to reduce chance of overflow.
    return np.exp(np.nanmean(np.log(x), axis=axis))


def scumble(labels: np.ndarray, reduce: bool = True) -> float:
    """Score  of  ConcUrrence  among  iMBalanced  LabEls (SCRUMBLE) from [1].

    Parameters
    ----------
    labels: np.ndarray
        Multi-labeled data that are dummy coded.

    reduce: bool = True
        Whether to provide the instance level scumble scores or to aggregate and create a single score.

    Returns
    -------
    scumble_score: float
        The average scumble across all instances.

    Examples
    --------
    >>> from torch_fairness.resampling import scumble
    >>> labels = np.array([
    >>>    [1, 0, 0, 1],
    >>>    [0, 1, 1, 0],
    >>>    [1, 0, 1, 0],
    >>>    [np.nan, np.nan, np.nan, np.nan],
    >>>    [1, 0, 0, 1],
    >>>    [0, 1, 0, 1],
    >>>    [np.nan, np.nan, 0, 1],
    >>>    [0, 1, np.nan, np.nan]
    >>> ]).astype(float)
    >>> print(scumble(labels))
    0.010168321420156588

    References
    ----------
    .. [1] Charte, F., Rivera, A. J., del Jesus, M. J., & Herrera, F. (2019). Dealing with difficult minority labels
            in imbalanced mutilabel data sets. Neurocomputing, 326, 39-53.
    """
    # TODO - Validate labels are dummy coded otherwise this is going to go poorly.
    irlb = np.nansum(labels, axis=0)
    irlb = irlb.max() / irlb
    scumble_score = labels.copy()
    scumble_score[scumble_score != 1.0] = np.nan
    scumble_score *= irlb
    # Since we are filtering out non-member labels by converting to nan and doing nan omission operations, this produces
    # a warning in samples that are either all missing or with partial missing labels and no membership. Temporarily
    # silence warning since this behavior is expected.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        scumble_score = 1 - (1 / np.nanmean(scumble_score, axis=1)) * geometric_mean(
            scumble_score
        )
    if reduce:
        scumble_score = np.nanmean(scumble_score)
    return scumble_score


def adjusted_hamming_distance(sample: np.ndarray, neighbors: np.ndarray) -> np.ndarray:
    """

    Parameters
    ----------
    sample: np.ndarray
        Reference dummy-coded multi-labeled data.
    neighbors: np.ndarray
        Comparison dummy-coded multi-labeled data.

    Returns
    -------
    np.ndarray
        Distances between [0,1] on degree of agreement between sample and neighbors controlling for
        missing data and the number of active labels.

    Notes
    -----
    The adjustment to hamming distance considers missing values in the sample and neighbors - only active labels
    that are being compared to non-missing values are considered when calculating.

    References
    ----------
    .. [1] Charte, F., Rivera, A. J., Jesus, M. J. D., & Herrera, F. (2014, September). MLeNN: a first approach to
        heuristic multilabel undersampling. In International Conference on Intelligent Data Engineering and
        Automated Learning (pp. 1-9). Springer, Cham.

    Examples
    --------
    >>> from torch_fairness.resampling import adjusted_hamming_distance
    >>> import numpy as np
    >>> sample = np.array([1., 0., 1.])
    >>> neighbors = np.array([[1., 0., 1.], [0., 1., 0.], [np.nan, np.nan, 1.], [1., 0., 0.]])
    >>> print(adjusted_hamming_distance(sample, neighbors))
    [0.         1.         0.33333333 0.33333333]
    """
    if sample.ndim != 1 or not isinstance(sample, np.ndarray):
        raise ValueError("Sample should be a 1D numpy array.")
    if neighbors.ndim != 2 or not isinstance(neighbors, np.ndarray):
        raise ValueError("Neighbors should be a 2D numpy array.")
    if sample.shape[0] != neighbors.shape[1]:
        raise ValueError(
            "Sample should have the same number of values as columns in neighbors."
        )
    if not all((sample != 0.0) | (sample != 1.0) | np.isnan(sample)):
        raise ValueError()
    _validate_dummy_coded_data(sample)
    _validate_dummy_coded_data(neighbors)
    # Adjust number of active labels by the presence of missing data in samples/neighbors. If a comparison with an
    # active label has a missing value then don't count active label.
    active_labels = neighbors.astype(bool) & ~np.isnan(neighbors) & ~np.isnan(sample)
    active_labels = active_labels.sum(1) + np.nansum(sample)
    cannot_compare = np.isnan(neighbors) | np.isnan(sample)
    hamming_distance = (sample != neighbors) & ~cannot_compare
    hamming_distance = hamming_distance.sum(1)
    return hamming_distance / active_labels


def get_sample_pool(
    sensitive: Union[pd.DataFrame, np.ndarray], idx: Optional[np.ndarray] = None
) -> List[Union[np.ndarray]]:
    """
    EXPLAIN ROLE OF IDX

    Args:
        sensitive:
        idx:

    Returns:

    """
    sample_pool = []
    dim_size = sensitive.ndim
    if dim_size not in [1, 2]:
        raise ValueError("Only 1D and 2D are supported.")
    if isinstance(sensitive, pd.DataFrame) or isinstance(sensitive, pd.Series):
        sensitive = sensitive.values
    sensitive_active = get_member(sensitive)
    if idx is None:
        idx = np.arange(sensitive.shape[0])
    if isinstance(sensitive, np.ndarray):
        if dim_size == 1:
            sample_idx = idx[sensitive_active]
            sample_pool.append(sample_idx)
        else:
            for col in range(sensitive.shape[1]):
                sample_idx = idx[sensitive_active[:, col]]
                sample_pool.append(sample_idx)
    else:
        raise ValueError(
            "Expected data to be one of the following: pd.DataFrame, pd.Series, torch.tensor, or np.array"
        )
    return sample_pool


class BaseResampler(ABC):
    """ """

    @abstractmethod
    def balance(
        self,
        sensitive: Union[np.ndarray, pd.DataFrame],
        *kwargs: Union[np.ndarray, pd.DataFrame],
    ) -> Dict[str, Union[np.ndarray, pd.DataFrame]]:
        ...

    @staticmethod
    def get_sample_size(sensitive: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Calculates the sample size"""
        sample_sizes = np.nansum(sensitive, axis=0)
        return sample_sizes

    def _validate_datatype(self, x: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if not (isinstance(x, pd.DataFrame) or isinstance(x, np.ndarray)):
            raise ValueError("Input was expected to be a pd.DataFrame or np.ndarray.")
        if x.ndim != 2:
            raise ValueError("Alll input must be 2D dimensional.")
        if isinstance(x, pd.DataFrame):
            return x.values
        else:
            return x


class MLROS(BaseResampler):
    """MultiLabel Random OverSampling (MLROS) from [1] - Uses information ratio to identify underrepresented group for
    oversampling.

    Parameters
    ----------
    sample_size_chunk: int = 5
        The number of samples to use in each update state as more samples are added. Higher leads to faster oversampling
        but can cause greater fluctuations in balancing groups.

    max_clone_percentage: float = 0.5
        The limit on how much of the dataset can be cloned. Once the threshold is reached oversampling stops, even if
        imbalance remains.

    random_state: int = None
        Random seed used to control sampling amoung neighbors.

    Attributes
    ----------
    sample_size_delta : int
        The change in the sample size: NEW_SIZE - OLD_SIZE

    added_index : List[int]
        The list of indices that were duplicated during oversampling.

    Notes
    -----


    Examples
    --------
    >>> from torch_fairness.resampling import MLROS
    >>> import pandas as pd
    >>> labels = pd.DataFrame([[1., 0.], [1., 0.], [1., 0.], [0., 1.]], columns=['Gender_Majority', 'Gender_Minority'])
    >>> resampler = MLROS(max_clone_percentage=0.5, random_state=1, sample_size_chunk=1)
    >>> new_data = resampler.balance(labels=labels, features=features)
    >>> print(new_data['labels'])
    array([[1., 0.],
           [1., 0.],
           [1., 0.],
           [0., 1.],
           [0., 1.],
           [0., 1.]])

    References
    ----------
    .. [1] Charte, F., Rivera, A. J., del Jesus, M. J., & Herrera, F. (2015). Addressing imbalance in multilabel
        classification: Measures and random resampling algorithms. Neurocomputing, 163, 3-16.
    """

    def __init__(
        self,
        sample_size_chunk: int = 5,
        max_clone_percentage: float = 0.5,
        random_state=None,
    ):
        if not (0 <= max_clone_percentage <= 1.0):
            raise ValueError("max_clone_percentage must be between 0 and 1.")
        if sample_size_chunk <= 0 or not isinstance(sample_size_chunk, int):
            raise ValueError("sample_size_chunk must be a positive non-zero integer.")
        self.max_clone_percentage = max_clone_percentage
        self.sample_size_chunk = sample_size_chunk
        self.random_state = set_random_state(random_state)

    def balance(
        self,
        labels: Union[pd.DataFrame, np.ndarray],
        **kwargs: Union[pd.DataFrame, np.ndarray],
    ) -> Dict[str, Union[pd.DataFrame, np.ndarray]]:
        """Reduce label imbalance using oversampling.

        Parameters
        ----------
        labels: pd.DataFrame
            Multi-labeled data that are dummy coded.

        Returns
        -------
        resampled_datasets: List[pd.DataFrame]
            The resampled datasets for all provided data.
        """
        _validate_dummy_coded_data(labels)
        data = {**{"labels": labels}, **kwargs}
        cols = {
            key: value.columns if isinstance(value, pd.DataFrame) else None
            for key, value in data.items()
        }
        data = {key: self._validate_datatype(value) for key, value in data.items()}

        max_clone_samples = int(data["labels"].shape[0] * self.max_clone_percentage)
        sample_sizes = self.get_sample_size(data["labels"])
        ir = imbalance_ratio(sample_sizes)
        mean_ir = ir.mean()
        sample_minority_groups = [
            group_i for group_i, group_ir in enumerate(ir) if group_ir > mean_ir
        ]

        # TODO - only calculate pool for groups of interest (sample_minority_groups)
        sample_pool = get_sample_pool(data["labels"])
        output_idx = np.arange(data["labels"].shape[0]).tolist()
        added_idx = []
        n_oversampled = 0
        for group_i in sample_minority_groups:
            group_pool = sample_pool[group_i]
            while (n_oversampled < max_clone_samples) and (ir[group_i] > mean_ir):
                resampled_idx = self.random_state.choice(
                    group_pool, self.sample_size_chunk, replace=False
                ).tolist()
                n_oversampled += len(resampled_idx)
                output_idx += resampled_idx
                added_idx += resampled_idx
                samples = data["labels"][resampled_idx, :]
                # Update information ratio
                sample_sizes += self.get_sample_size(samples)
                ir = imbalance_ratio(sample_sizes)
                mean_ir = ir.mean()
            if n_oversampled >= max_clone_samples:
                break
        out = {}
        for name, data_i in data.items():
            resampled_data = data_i[output_idx, :]
            col_i = cols[name]
            if col_i is not None:
                resampled_data = pd.DataFrame(resampled_data, columns=col_i)
            out[name] = resampled_data
        # Attributes
        self.sample_size_delta = len(output_idx) - data["labels"].shape[0]
        self.added_index = added_idx
        return out


class MLRUS(BaseResampler):
    """Multi-Label Random UnderSampling (MLRUS) from [1].

    Parameters
    ----------
    random_state: int = None
        Random seed used to control sampling amoung neighbors.

    Attributes
    ----------
    sample_size_delta : int
        The change in the sample size: NEW_SIZE - OLD_SIZE

    removed_index : List[int]
        The list of indices that were removed during undersampling.

    Notes
    -----
    This allows duplicate synthetic examples as it always creates synthetic from real samples in order to prevent
    retraining the KNN each time a new sample is generated.

    Examples
    --------
    >>> from torch_fairness.resampling import MLRUS
    >>> import pandas as pd
    >>> labels = pd.DataFrame([[1., 0.], [1., 0.], [1., 0.], [0., 1.]], columns=['Gender_Majority', 'Gender_Minority'])
    >>> resampler = MLRUS(random_state=1, sample_size_chunk=1)
    >>> new_data = resampler.balance(labels=labels.values)
    >>> print(new_data['labels'])
        [[1. 0.]
         [0. 1.]]

    References
    ----------
    .. [1] Charte, F., Rivera, A. J., del Jesus, M. J., & Herrera, F. (2015). Addressing imbalance in multilabel
        classification: Measures and random resampling algorithms. Neurocomputing, 163, 3-16.
    """

    def __init__(self, sample_size_chunk: int = 1, random_state=None):
        if (not isinstance(sample_size_chunk, int)) or sample_size_chunk <= 0:
            raise ValueError()
        self.sample_size_chunk = sample_size_chunk
        self.random_state = set_random_state(random_state=random_state)

    def balance(
        self,
        labels: Union[pd.DataFrame, np.ndarray],
        **kwargs: Union[pd.DataFrame, np.ndarray],
    ) -> Dict[str, Union[pd.DataFrame, np.ndarray]]:
        """Reduce label imbalance using undersampling.

        Parameters
        ----------
        labels: pd.DataFrame
            Multi-labeled data that are dummy coded.

        Returns
        -------
        resampled_datasets: List[pd.DataFrame]
            The resampled datasets for all provided data.
        """
        _validate_dummy_coded_data(labels)
        data = {**{"labels": labels}, **kwargs}
        cols = {
            key: value.columns if isinstance(value, pd.DataFrame) else None
            for key, value in data.items()
        }
        data = {key: self._validate_datatype(value) for key, value in data.items()}

        sample_sizes = self.get_sample_size(data["labels"])
        ir = imbalance_ratio(sample_sizes)
        mean_ir = ir.mean()

        sample_majority_groups = [
            group_i for group_i, group_ir in enumerate(ir) if group_ir < mean_ir
        ]
        sample_minority_groups = [
            group_i for group_i, group_ir in enumerate(ir) if group_ir >= mean_ir
        ]
        sample_pool = get_sample_pool(data["labels"])

        # Get majority sample pool - don't remove any samples in a minority group
        all_minority_idx = np.concatenate(
            [sample_pool[i] for i in sample_minority_groups], axis=0
        )
        majority_only_idx = np.setdiff1d(
            np.arange(data["labels"].shape[0]), all_minority_idx
        )
        majority_sample_pool = [sample_pool[i] for i in sample_majority_groups]
        majority_sample_pool = [
            [i for i in attribute if i in majority_only_idx]
            for attribute in majority_sample_pool
        ]
        all_removal_idx = []
        for group_i, group_pool in zip(sample_majority_groups, majority_sample_pool):
            group_pool = np.setdiff1d(group_pool, all_removal_idx)
            self.random_state.shuffle(group_pool)
            while ir[group_i] < mean_ir and group_pool.shape[0] > 0:
                remove_idx = self.random_state.choice(
                    group_pool, self.sample_size_chunk, replace=False
                ).tolist()
                all_removal_idx += remove_idx
                # Remove selected IDX from original using view
                group_pool = np.setdiff1d(group_pool, remove_idx)
                # Update information ratio
                samples = data["labels"][remove_idx, :]
                sample_sizes -= self.get_sample_size(samples)
                ir = imbalance_ratio(sample_sizes)
                mean_ir = ir.mean()
            if group_pool.shape[0] == 0 and ir[group_i] < mean_ir:
                warnings.warn(
                    "Ran out of samples to remove for group - continuing to next group.",
                    UserWarning,
                )
        output_idx = np.arange(data["labels"].shape[0])
        output_idx = np.setdiff1d(output_idx, all_removal_idx).tolist()
        out = {}
        for name, data_i in data.items():
            resampled_data = data_i[output_idx, :]
            col = cols[name]
            if col is not None:
                resampled_data = pd.DataFrame(resampled_data, columns=col)
            out[name] = resampled_data
        self.sample_size_delta = len(output_idx) - data["labels"].shape[0]
        self.removed_index = all_removal_idx
        return out


class MLSMOTE(BaseResampler):
    """Multi-label Synthetic Minority Oversampling Technique (MLSMOTE) from [1].

    Handles both categorical and continuous variables. Categorical variables are generated by taking the mode of the
    nearest K-neighbors. Continuous variables are generated by interpolating between a reference sample and a
    random selection of one of the k-nearest neighbors.

    Parameters
    ----------
    k_neighbors: int = 5
        The number of neighbors to be used to generating the synthetic sample.

    random_state: int = None
        Random seed used to control sampling amoung neighbors.

    discrete_options_cutoff: int = 20
        The number of unique values used to determine if a columns is discrete of continuous - used to select method of
        generating synthetic sample.

    Attributes
    ----------
    sample_size_delta : int
        The change in the sample size: NEW_SIZE - OLD_SIZE

    synthetic_index : List[int]
        The list of indices that were synthesized.

    Notes
    -----
    This allows duplicate synthetic examples as it always creates synthetic from real samples in order to prevent
    retraining the KNN each time a new sample is generated.

    Examples
    --------
    >>> from torch_fairness.resampling import MLSMOTE
    >>> import pandas as pd
    >>> labels = pd.DataFrame([[1., 0.], [1., 0.], [1., 0.], [0., 1.], [0., 1.]], columns=['Gender_Majority', 'Gender_Minority'])
    >>> features = pd.DataFrame([[0.5, 0.2], [0.2, 0.1], [0., 0.8], [0.1, 0.1], [1., 1.2]], columns=['feature_a', 'feature_b'])
    >>> resampler = MLSMOTE(random_state=1, k_neighbors=1)
    >>> new_data = resampler.balance(labels=labels.values, features=features.values)
    >>> print(new_data['labels'])
        [[0. 1.]
         [1. 0.]
         [1. 0.]
         [1. 0.]
         [0. 1.]
         [0. 1.]]

    References
    ----------
    .. [1] Charte, F., Rivera, A. J., del Jesus, M. J., & Herrera, F. (2015). MLSMOTE: Approaching imbalanced
        multilabel learning through synthetic instance generation. Knowledge-Based Systems, 89, 385-397.
    """

    def __init__(
        self,
        k_neighbors: int = 5,
        random_state: Union[int, None, np.random.RandomState] = None,
        max_clone_percentage: float = 0.5,
        discrete_options_cutoff: int = 20,
    ):
        self.random_state = set_random_state(random_state=random_state)
        self.max_clone_percentage = max_clone_percentage
        self.k_neighbors = k_neighbors
        self.discrete_options_cutoff = discrete_options_cutoff
        self.sample_size_delta = None
        self.synthetic_index = None

    def balance(
        self,
        labels: Union[pd.DataFrame, np.ndarray],
        features: Union[pd.DataFrame, np.ndarray],
        **kwargs: Union[pd.DataFrame, np.ndarray],
    ) -> Dict[str, Union[pd.DataFrame, np.ndarray]]:
        """Reduce label imbalance using synthetic oversampling.

        Parameters
        ----------
        labels: pd.DataFrame, np.ndarray
            Multi-labeled data that are dummy coded.

        features: pd.DataFrame, np.ndarray
            Features used to find nearest neighbors for each sample.

        Returns
        -------
        resampled_datasets: Dict[str, Union[pd.DataFrame, np.ndarray]]
            The resampled datasets for all provided data.
        """
        _validate_dummy_coded_data(labels)
        data = {**{"labels": labels, "features": features}, **kwargs}
        cols = {
            key: value.columns if isinstance(value, pd.DataFrame) else None
            for key, value in data.items()
        }
        data = {key: self._validate_datatype(value) for key, value in data.items()}
        if np.isnan(data["features"]).sum() != 0:
            raise ValueError(
                "Input features contains NaN - cannot calculate neighbors."
            )

        max_clone_samples = int(data["labels"].shape[0] * self.max_clone_percentage)
        sample_sizes = self.get_sample_size(data["labels"])
        ir = imbalance_ratio(sample_sizes)
        mean_ir = ir.mean()
        sample_minority_groups = [
            group_i for group_i, group_ir in enumerate(ir) if group_ir > mean_ir
        ]
        sample_pool = get_sample_pool(data["labels"])
        if np.min([len(sample_pool[i]) for i in sample_minority_groups]) < (
            self.k_neighbors + 1
        ):
            raise ValueError(
                "All groups must have minimum sample size larger than k_neighbors"
            )
        n_oversampled = 0
        new_data = {name: [] for name in data.keys()}
        columns_are_continuous = {
            name: self._infer_if_continuous(values) for name, values in data.items()
        }
        for group_i in sample_minority_groups:
            group_pool = sample_pool[group_i]
            # Shuffle pool in-case of ordering
            self.random_state.shuffle(group_pool)
            group_feature_data = data["features"][group_pool, :]
            # Initialize nearest neighbors using group samples
            kdt = KDTree(group_feature_data)
            sample_i = 0
            while (
                (n_oversampled < max_clone_samples)
                and (ir[group_i] > mean_ir)
                and (sample_i < len(group_pool))
            ):
                n_oversampled += 1
                sample_idx = group_pool[sample_i]
                _, neighbor_idx = kdt.query(
                    x=data["features"][sample_idx, :], k=self.k_neighbors + 1
                )
                neighbor_idx = neighbor_idx[
                    1:
                ]  # Remove sample_idx which is the smallest distance to itself
                neighbor_idx = group_pool[neighbor_idx]
                reference_idx = self.random_state.choice(neighbor_idx)
                # Create new sample along all attributes (e.g., features, labels, labels, ...)
                for name, data_i in data.items():
                    new_sample_data_i = self._create_sample(
                        starting_sample=data_i[sample_idx, :],
                        reference_sample=data_i[reference_idx, :],
                        neighbors=data_i[neighbor_idx, :],
                        is_continuous=columns_are_continuous[name],
                    )
                    new_data[name].append(new_sample_data_i)
                    if name == "labels":
                        sample_sizes += new_sample_data_i
                        ir = imbalance_ratio(sample_sizes)
                        mean_ir = ir.mean()
                sample_i += 1
            if n_oversampled >= max_clone_samples:
                break
        # Combine new samples w/ dataframes
        out = {}
        for name in data.keys():
            new_data_i = np.concatenate([new_data[name], data[name]], axis=0)
            col = cols[name]
            if col is not None:
                new_data_i = pd.DataFrame(new_data_i, columns=col)
            out[name] = new_data_i

        # Attributes
        self.sample_size_delta = n_oversampled - data["labels"].shape[0]
        self.synthetic_index = list(
            range(data["labels"].shape[0], data["labels"].shape[0] + n_oversampled)
        )
        return out

    def _infer_if_continuous(self, x: np.ndarray) -> np.ndarray:
        """Infers for each column if it is continuous or not. Used to select method of creating synthetic samples."""
        n_unique_values = np.array([len(np.unique(i)) for i in x.T])
        is_numeric = np.array([is_numeric_dtype(i) for i in x.T])
        is_continuous = (n_unique_values >= self.discrete_options_cutoff) & is_numeric
        return is_continuous

    def _create_sample(
        self,
        starting_sample: np.ndarray,
        reference_sample: np.ndarray,
        neighbors: np.ndarray,
        is_continuous: np.ndarray,
    ) -> np.ndarray:
        """Creates a synthetic sample between the starting and reference sample. Categorical data are created based on
        the mode of the neighbors and continuous data are interpreted between two samples.
        """
        is_discrete = ~is_continuous
        new_sample = np.zeros(starting_sample.shape[0])
        # Create continuous data
        new_sample[is_continuous] = self._create_continuous_sample(
            starting_sample=starting_sample[is_continuous],
            reference_sample=reference_sample[is_continuous],
        )
        # Create discrete data
        new_sample[is_discrete] = self._create_discrete_sample(
            neighbors=neighbors[:, is_discrete]
        )
        return new_sample

    @staticmethod
    def _create_discrete_sample(neighbors: np.ndarray) -> np.ndarray:
        """Creates a new discrete sample based off of the mode of the neighbors."""
        new_sample = mode(neighbors, keepdims=False, axis=0, nan_policy="omit")
        return new_sample.mode

    def _create_continuous_sample(
        self, starting_sample: np.ndarray, reference_sample: np.ndarray
    ) -> np.ndarray:
        """Creates continuous sample based on interpolated distance between two samples."""
        include_diff = self.random_state.choice([0, 1], starting_sample.shape[0])
        offset = reference_sample - starting_sample
        offset[np.isnan(offset)] = 0
        new_sample = starting_sample + offset * include_diff
        return new_sample


class MLeNN(BaseResampler):
    """Multi-label edited nearest neighbor (MLeNN) from [1]. Undersampling strategy based on calculating the hamming
    distance between a sample and its k-nearest neighbors and removing it if it has exceeded a user-specified threshold.


    Parameters
    ----------
    k_neighbors: int = 5
        The number of neighbors to be used to generating the synthetic sample.

    neighbor_threshold: float
        Threshold used to determine removing sample based on average distance to neighbors using an adjusted hamming
        distance.

    random_state: int = None
        Random seed used for reproducing stochastic operations.

    Attributes
    ----------
    sample_size_delta : int
        The change in the sample size: NEW_SIZE - OLD_SIZE

    removed_index : List[int]
        The list of indices that were removed during undersampling.

    Notes
    -----
    Hamming distance has been adjusted to account for different numbers of active labels to create a consistent
    interpretation of the distance.

    Examples
    --------
    >>> from torch_fairness.resampling import MLeNN
    >>> import pandas as pd
    >>> labels = pd.DataFrame([[1., 0.], [1., 0.], [1., 0.], [0., 1.], [0., 1.]], columns=['Gender_Majority', 'Gender_Minority'])
    >>> features = pd.DataFrame([[0.5, 0.2], [0.2, 0.1], [0., 0.8], [0.1, 0.1], [1., 1.2]], columns=['feature_a', 'feature_b'])
    >>> resampler = MLeNN(random_state=1, k_neighbors=1)
    >>> new_data = resampler.balance(labels=labels.values, features=features.values)
    >>> print(new_data['labels'])
        [[1. 0.]
         [1. 0.]
         [0. 1.]
         [0. 1.]]

    References
    ----------
    .. [1] Charte, F., Rivera, A. J., Jesus, M. J. D., & Herrera, F. (2014, September). MLeNN: a first approach to
            heuristic multilabel undersampling. In International Conference on Intelligent Data Engineering and
            Automated Learning (pp. 1-9). Springer, Cham.
    """

    def __init__(
        self, k_neighbors: int = 3, neighbor_threshold: float = 0.5, random_state=None
    ):
        if (not isinstance(k_neighbors, int)) or k_neighbors <= 0:
            raise ValueError("k_neighbors must be a positive non-zero integer.")
        if (
            (not isinstance(neighbor_threshold, float))
            or neighbor_threshold <= 0
            or neighbor_threshold >= 1
        ):
            raise ValueError("neighbor_threshold must be a float between 0 and 1.")
        self.k_neighbors = k_neighbors
        self.neighbor_threshold = neighbor_threshold
        self.random_state = set_random_state(random_state=random_state)

    @staticmethod
    def _remove_from_pool(
        sample_pool: List[np.array], idx: List[int]
    ) -> List[np.array]:
        for col_i in range(len(sample_pool)):
            sample_pool[col_i] = np.setdiff1d(sample_pool[col_i], idx)
        return sample_pool

    def balance(
        self,
        labels: Union[pd.DataFrame, np.ndarray],
        features: Union[pd.DataFrame, np.ndarray],
        **kwargs: List[Union[pd.DataFrame, np.ndarray]],
    ) -> Dict[str, Union[pd.DataFrame, np.ndarray]]:
        """Reduce label imbalance using undersampling.

        Parameters
        ----------
        labels: pd.DataFrame, np.ndarray
            Multi-labeled data that are dummy coded.

        features: pd.DataFrame, np.ndarray
            Features used to find nearest neighbors for each sample.

        Returns
        -------
        resampled_datasets: Dict[str, Union[pd.DataFrame, np.ndarray]]
            The resampled datasets for all provided data.
        """
        _validate_dummy_coded_data(labels)
        data = {**{"labels": labels, "features": features}, **kwargs}
        cols = {
            key: value.columns if isinstance(value, pd.DataFrame) else None
            for key, value in data.items()
        }
        data = {key: self._validate_datatype(value) for key, value in data.items()}
        if np.isnan(data["features"]).sum() != 0:
            raise ValueError(
                "Input features contains NaN - cannot calculate neighbors."
            )
        if self.k_neighbors >= labels.shape[0]:
            raise ValueError("The number of samples must be larger than k_neighbors.")
        sample_sizes = self.get_sample_size(data["labels"])
        ir = imbalance_ratio(sample_sizes)
        mean_ir = ir.mean()

        sample_majority_groups = [
            group_i for group_i, group_ir in enumerate(ir) if group_ir < mean_ir
        ]
        sample_minority_groups = [
            group_i for group_i, group_ir in enumerate(ir) if group_ir >= mean_ir
        ]
        sample_pool = get_sample_pool(data["labels"])

        # Get majority sample pool - don't remove any samples in a minority group
        minority_idx = np.concatenate(
            [sample_pool[i] for i in sample_minority_groups], axis=0
        )
        majority_only_idx = np.setdiff1d(
            np.arange(data["labels"].shape[0]), minority_idx
        ).tolist()
        majority_sample_pool = get_sample_pool(
            data["labels"][majority_only_idx, :], idx=np.array(majority_only_idx)
        )
        all_removal_idx = []
        kdt = KDTree(data=data["features"])
        for group_i in sample_majority_groups:
            group_pool = majority_sample_pool[group_i]
            self.random_state.shuffle(group_pool)
            for sample_idx in group_pool:
                if ir[group_i] >= mean_ir:
                    break
                _, neighbor_idx = kdt.query(
                    data["features"][sample_idx, :], k=self.k_neighbors + 1
                )
                neighbor_idx = neighbor_idx[1:]
                ahd = adjusted_hamming_distance(
                    data["labels"][sample_idx, :], data["labels"][neighbor_idx, :]
                )
                if ahd.mean() >= self.neighbor_threshold:
                    remove_idx = [sample_idx]
                    all_removal_idx += remove_idx

                    # Remove selected IDX from original using view
                    group_pool = np.setdiff1d(group_pool, remove_idx)
                    majority_sample_pool = self._remove_from_pool(
                        majority_sample_pool, remove_idx
                    )

                    # Update information ratio
                    sample_sizes -= self.get_sample_size(data["labels"][remove_idx, :])
                    ir = imbalance_ratio(sample_sizes)
                    mean_ir = ir.mean()
        output_idx = np.arange(data["labels"].shape[0])
        output_idx = np.setdiff1d(output_idx, all_removal_idx).tolist()
        out = {}
        for name, data_i in data.items():
            col = cols[name]
            data_i = data_i[output_idx, :]
            if col is not None:
                data_i = pd.DataFrame(data_i, columns=col)
            out[name] = data_i
        self.sample_size_delta = len(output_idx) - data["labels"].shape[0]
        self.removed_index = all_removal_idx
        return out
