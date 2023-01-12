
from typing import Union

import numpy as np
import pandas as pd


def set_random_state(random_state: Union[None, int, np.random.RandomState]) -> np.random.RandomState:
    """Sets the random state to ensure reproducible results.

    Parameters
    ----------
    random_state: None, np.random.RandomState
        If seed is None then randomly initialize a RandomState - otherwise use the provided state.

    Returns
    -------
    random_state: np.random.RandomState
        RandomState object used for stochastic operations.
    """
    if random_state is None:
        return np.random.mtrand._rand
    elif isinstance(random_state, int):
        return np.random.RandomState(seed=random_state)
    elif isinstance(random_state, np.random.RandomState):
        return random_state
    else:
        raise ValueError(f"{random_state} cannot be used to set a random state.")


def _validate_dummy_coded_data(x: Union[np.ndarray, pd.DataFrame]) -> None:
    if not np.all((x == 0.) | (x == 1.) | np.isnan(x)):
        raise ValueError('Found unexpected values in dummy coded array: should only have {0, 1, np.nan}')
