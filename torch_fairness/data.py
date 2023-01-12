
from collections import Counter
from typing import List, Union
from dataclasses import dataclass
import numbers

import numpy as np
import pandas as pd
import torch

from torch_fairness.exceptions import NotFittedError
from torch_fairness.exceptions import MissingSensitiveGroupsError
from torch_fairness.exceptions import DummyCodingError


def get_member(sensitive: torch.tensor, x: torch.tensor = None) -> torch.tensor:
    """Identifies the active member instances within a tensor. If a second tensor is specified, it is indexed by the
    selected indices.

    Parameters
    ----------
    sensitive: torch.tensor
        Sensitive data that has been dummy coded where 1. means member of a group.
    x: torch.tensor, None
        Optional tensor to be indexed by active members of sensitive.

    Returns
    -------
    torch.tensor
        Returns a boolean if x is not provided, or the indexed values of x by sensitive if it is provided.

    Examples
    --------
    >>> from torch_fairness.data import get_member
    >>> import torch
    >>> sensitive = torch.tensor([1, 0, 1, 0])
    >>> print(get_member(sensitive=sensitive))
    tensor([ True, False,  True, False])
    >>> x = torch.tensor([1, 2, 3, 4])
    >>> print(get_member(sensitive=sensitive, x=x))
    tensor([1, 3])
    """
    if x is not None:
        if x.shape[0] != sensitive.shape[0]:
            raise ValueError("X and sensitive must be the same shape.")
        if x.ndim == 1:
            return x[sensitive == 1]
        elif x.ndim == 2:
            return x[sensitive == 1, :]
        else:
            raise ValueError("X must be either 1D or 2D.")
    else:
        return sensitive == 1


@dataclass(frozen=True)
class SensitiveAttribute:
    """Data class holding information about a sensitive attribute. Used to construct 'SensitiveMap'.

    Parameters
    ----------

    majority: int, str
        The name of the majority group (e.g., white).
    minority: list[Union[int,str]]
        List of the names of the minority grousp in the sensitive attribute (e.g., black, asian).
    name: int, str
        The name of the sensitive attribute (e.g., race, gender).

    Examples
    --------
    >>> from torch_fairness.data import SensitiveAttribute
    >>> print(SensitiveAttribute(name='Gender', minority=['Female'], majority='Male'))
    SensitiveAttribute(majority='Male', minority=['Female'], name='Gender')
    """
    majority: Union[int, str]
    minority: List[Union[int, str]]
    name: Union[str, int]


class SensitiveMap:
    """Class containing data on the sensitive attributes of interest. Used to simplify transformations between
    categorical and dummy coded representation of sensitive data as well as tracking which columns in the dummy
    coded sensitive are majority/minority pairs for calculation in loss functions.


    Parameters
    ----------

    *sensitive_attributes: SensitiveAttribute, Dict
        Inputs are either SensitiveAttributes or Dicts with the necessary keys to construct SensitiveAttribute (e.g.,
        {'name': 'Race', 'minority': ['Black', 'Asian'], 'majority': 'White'})

    Attributes
    ----------

    majority_minority_pairs: list[int, list[int]]
        List containing the (majority_idx, [minority_idx]) pairs for idx in

    group_names: list[str]
        The names of the groups included across all sensitive attributes.

    Notes
    -----


    Examples
    --------

    SensitiveMap can either be inferred from input data or specified manually.

    >>> from torch_fairness.data import SensitiveMap, SensitiveAttribute
    >>> sensitive_map = SensitiveMap(SensitiveAttribute(majority='Male', minority=['Female'], name='Gender'))
    >>> print(SensitiveMap)
    [SensitiveAttribute(majority='Male', minority=['Female'], name='Gender')]
    >>> import pandas as pd
    >>> data = pd.DataFrame({'Gender': ['Male', 'Male', 'Female']})
    >>> print(SensitiveMap.infer(data, minimum_sample_size=1))
    [SensitiveAttribute(majority='Male', minority=['Female'], name='Gender')]

    """
    def __init__(self, *args: Union[dict, SensitiveAttribute]):
        attributes = []
        for i in args:
            if isinstance(i, dict):
                i = SensitiveAttribute(**i)
            elif not isinstance(i, SensitiveAttribute):
                raise ValueError("Expected args to be either a dict or SensitiveAttribute.")
            if i.majority in i.minority:
                raise ValueError(f"SensitiveAttribute {i.name} cannot have majority group in minority groups.")
            attributes.append(i)
        self._attributes = attributes

        self.majority_minority_pairs = []
        idx_counter = 0
        for attribute in self._attributes:
            n_groups = 1 + len(attribute.minority)
            group_idx = list(range(idx_counter, idx_counter + n_groups))
            idx_counter += n_groups
            self.majority_minority_pairs.append((group_idx[0], tuple(group_idx[1:])))

        self.group_names = []
        for attribute in self._attributes:
            self.group_names.append(attribute.majority)
            self.group_names += attribute.minority

    def __getitem__(self, item: int) -> SensitiveAttribute:
        return self._attributes[item]

    def __repr__(self):
        return str(self._attributes)

    def __len__(self):
        return len(self._attributes)

    def __eq__(self, other):
        if isinstance(other, SensitiveMap):
            return self._attributes == other._attributes
        return False

    @classmethod
    def infer(cls, x: Union[pd.DataFrame, np.ndarray], minimum_sample_size: int = 30):
        """Infer the SensitiveMap based on the provided sensitive data (not dummy coded) and a specified
        minimum smaple size.

        Parameters
        ----------
        x: pd.DataFrame, np.ndarray
            The sensitive attributes of interest structured as categorical variables.
        minimum_sample_size: int, default=30
            The minimum sample size for inclusion of a group within each sensitive attribute.

        Returns
        -------
        self: SensitiveMap
            Inferred sensitive attributes and sufficiently represented groups.
        """
        if isinstance(x, pd.DataFrame):
            colnames = x.columns
            x = x.values
        elif isinstance(x, np.ndarray):
            colnames = [i for i in range(x.shape[1])]
        else:
            raise ValueError("Data should be a np.ndarray or pd.DataFrame.")
        sensitive_map = []
        for col_i, colname in enumerate(colnames):
            attribute_data = x[:, col_i]
            attribute_data = attribute_data[~pd.isnull(attribute_data)]
            counts = Counter(attribute_data)
            sufficient_counts = {key: value for key, value in counts.items() if value >= minimum_sample_size}
            if len(sufficient_counts) < 2:
                raise ValueError(f"{col_i} column has less than two groups satisfying minimum sample size "
                                 f"{minimum_sample_size}. At least two groups are needed to form a majority and minority "
                                 f"group.")
            groups = list(sufficient_counts.keys())
            majority = groups[0]
            minority = groups[1:]
            group = SensitiveAttribute(majority=majority, minority=minority, name=colname)
            sensitive_map.append(group)
        sensitive_map = cls(*sensitive_map)
        return sensitive_map


class SensitiveTransformer:
    """Transformer between categorical and dummy coded representations of sensitive data.

    Parameters
    ----------
    sensitive_map: SensitiveMap, None
        Input data.
    minimum_sample_size: int, default=30
        The minimum sample size used when inferring SensitiveMap if not provided.

    Examples
    --------
    >>> from torch_fairness.data import SensitiveTransformer
    >>> import pandas as pd
    >>> data = pd.DataFrame({"gender": ['male', 'female', 'male', 'male', 'female', 'nonbinary', float('nan')]})
    >>> transformer = SensitiveTransformer(minimum_sample_size=2)
    >>> print(transformer.fit_transform(data))
    [[ 1.  0.]
     [ 0.  1.]
     [ 1.  0.]
     [ 1.  0.]
     [ 0.  1.]
     [ 0.  0.]
     [nan nan]]
    """
    _trained: bool = False

    def __init__(self, sensitive_map: SensitiveMap = None, minimum_sample_size: int = 30):
        if not isinstance(minimum_sample_size, int) or minimum_sample_size <= 0:
            raise ValueError("minimum_sample_size should be a positive non-zero integer.")
        if sensitive_map is not None:
            if not isinstance(sensitive_map, SensitiveMap):
                raise ValueError('sensitive_map should be a SensitiveMap().')
        self.sensitive_map = sensitive_map
        self.minimum_sample_size = minimum_sample_size

    def fit(self, x: Union[pd.DataFrame, np.ndarray]):
        """Create and validate SensitiveMap for transformation.

        Parameters
        ----------
        x: pd.DataFrame, np.ndarray
            Input data.

        Returns
        -------
        self: object
            Fitted transformer.
        """
        if self.sensitive_map is None:
            # Infer sensitive group using minimum sample size
            self.sensitive_map = SensitiveMap.infer(x, minimum_sample_size=self.minimum_sample_size)
        else:
            self._validate_data(x, is_dummy_coded=False)
            x = self._convert_to_array(x)
            # Ensure that all groups that are specified are present and in the minimum sample size
            for col_i, sensitive_attribute in enumerate(self.sensitive_map):
                groups = [sensitive_attribute.majority, *sensitive_attribute.minority]
                group_counts = Counter(x[:, col_i])
                # Check if expected groups are present
                group_diff = set(groups) - set(group_counts.keys())
                if len(group_diff) != 0:
                    raise ValueError(f"Expected groups for {sensitive_attribute.name} not found - {group_diff}")
                matched_groups = [i for i in groups if group_counts.get(i, 0) >= self.minimum_sample_size]
                if len(matched_groups) != len(groups):
                    raise ValueError(f"Groups failed to meet minimum_sample_size {self.minimum_sample_size}: "
                                     f"{set(groups) - set(matched_groups)}")

        self._sensitive_dtype = []
        for col_i, sensitive_attribute in enumerate(self.sensitive_map):
            groups = [sensitive_attribute.majority, *sensitive_attribute.minority]
            if all([isinstance(i, numbers.Number) for i in groups]):
                self._sensitive_dtype.append(np.float32)
            else:
                self._sensitive_dtype.append(object)
        self._trained = True
        return self

    def transform(self, x: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Transform categorical group data into dummy coded majority/minority groups.

        Parameters
        ----------
        x: pd.DataFrame, np.ndarray
            Input data.

        Returns
        -------
        x_tr: pd.DataFrame, np.ndarray
            Transformed data.
        """
        self._validate_trained()
        self._validate_data(x, is_dummy_coded=False)
        x = self._convert_to_array(x)
        out = []
        for col_i, sensitive_attribute in enumerate(self.sensitive_map):
            attribute_data = x[:, col_i]
            temp = []
            is_missing = pd.isnull(attribute_data)
            group_values = [sensitive_attribute.majority, *sensitive_attribute.minority]
            for group in group_values:
                temp.append(attribute_data == group)
            temp = np.column_stack(temp).astype(np.float32)
            temp[is_missing] = np.nan
            out.append(temp)
        out = np.column_stack(out)
        return out

    def fit_transform(self, x: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Fit to data, then transform it.

        Parameters
        ----------
        x: pd.DataFrame, np.ndarray
            Input data.

        Returns
        -------
        x_tr: pd.DataFrame, np.ndarray
            Transformed data.
        """
        self.fit(x)
        out = self.transform(x)
        return out

    def inverse_transform(self, x: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """Scale back the data to the original categorical representation.

        Parameters
        ----------
        x: pd.DataFrame, np.ndarray
            Already transformed data to be returned to original representation.

        Returns
        -------
        x_tr: pd.DataFrame, np.ndarray
            Transformed data.
        """
        self._validate_trained()
        self._validate_data(x, is_dummy_coded=True)
        x = self._convert_to_array(x)
        out = []
        all_group_names = self.sensitive_map.group_names
        for i, (maj, min) in enumerate(self.sensitive_map.majority_minority_pairs):
            combined_idx = [maj, *min]
            group_names = all_group_names[combined_idx[0]:(combined_idx[-1] + 1)]
            group_names = {float(i): j for i, j in enumerate(group_names)}
            missing = np.isnan(x[:, combined_idx])
            all_missing = missing.all(axis=1)
            if (all_missing != missing.any(axis=1)).any():
                raise DummyCodingError(f"Partial missing was observed for {self.sensitive_map[i].name}. If one group is missing they all should be.")
            if (np.nan_to_num(x[:, combined_idx].sum(axis=1), nan=1) != 1).any():
                raise MissingSensitiveGroupsError(f"Multiple group membership within label {self.sensitive_map[i].name} were found - groups should be non-overlapping within an attribute.")
            group = x[:, combined_idx].argmax(1).astype(np.float32)
            group = np.array([group_names[x] for x in group], dtype=self._sensitive_dtype[i])
            group[all_missing] = np.nan
            out.append(group)
        out = np.column_stack(out)
        return out

    def _convert_to_array(self, x: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if isinstance(x, pd.DataFrame):
            x = x.values
        return x

    def _validate_data(self, x: Union[pd.DataFrame, np.ndarray], is_dummy_coded: bool) -> None:
        self._validate_datatype(x, is_dummy_coded)
        self._validate_groups(x, is_dummy_coded)

    def _validate_datatype(self, x: Union[pd.DataFrame, np.ndarray], is_dummy_coded: bool) -> None:
        if not (isinstance(x, pd.DataFrame) or isinstance(x, np.ndarray)):
            raise ValueError("Data was expected to be a pd.DataFrame or np.ndarray.")
        if is_dummy_coded:
            # Data should only be 0., 1., or np.nan after dummy coding
            correct = np.all((x == 0.) | (x == 1.) | pd.isnull(x))
            if not correct:
                raise DummyCodingError("All values should be either 1., 0., or nan - encountered unexpected value.")

    # TODO - This looks confusing and redundant
    def _validate_groups(self, x: Union[pd.DataFrame, np.ndarray], is_dummy_coded: bool) -> None:
        if is_dummy_coded:
            if x.shape[1] != len(self.sensitive_map.group_names):
                # TODO - Use custom exception and write better error message
                if len(self.sensitive_map) != x.shape[1]:
                    raise ValueError(
                        "The number of sensitive groups in sensitive_map does not match the number of input columns.")
        else:
            # Ensure the number and order of attributes of data matches the sensitive_map
            if isinstance(x, pd.DataFrame):
                observed_col = x.columns.tolist()
                expected_col = [sensitive_attribute.name for sensitive_attribute in self.sensitive_map]
                if observed_col != expected_col:
                    raise ValueError(f"Expected columns ordered as: {expected_col} - instead received {observed_col}")
            else:
                n_observed_col = x.shape[1]
                n_expected_col = len(self.sensitive_map)
                if n_expected_col != n_expected_col:
                    raise ValueError(f"Expected {n_expected_col} attributes but found {n_observed_col}.")

    def _validate_trained(self):
        if not self._trained:
            raise NotFittedError(f"{self.__class__.__name__} is not fitted yet. Call 'fit' before using this method.")

