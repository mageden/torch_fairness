
from typing import Iterator, List
import warnings
import math

import pandas as pd
import torch
import numpy as np
from torch.utils.data import Sampler

from torch_fairness.resampling import get_sample_pool
from torch_fairness.util import set_random_state


class MLStratifiedBatchSampler(Sampler[List[int]]):
    """Multi-label stratified batch sampler based on marginal random sampling.

    Each batch is broken into a number of chunks equal to the number of classes. These chunks then randomly sample
    observations from the smallest sample size label, guaranteeing a minimum size equal to the chunk size
    (batch_size//num_chunks).

    Parameters
    ----------
    labels: torch.tensor
        Dummy coded multi-label data (e.g., sensitive groups). Used to identify which indices to include in each batch
        to ensure all members are present.

    batch_size: int
        Number of samples to include in a batch.

    limit_replacement: bool = False
        Whether to preferentially sample observations that haven't been observed within an epoch - if False then sample
        with replacement.

    random_state: int = None
        Random seed used for reproducing stochastic operations.


    Examples
    --------
    >>> from torch_fairness.sampler import MLStratifiedBatchSampler
    >>> import torch
    >>> from torch.utils.data import TensorDataset, DataLoader
    >>> labels = torch.tensor([[0, 1], [1, 0], [1, 1], [0, 1]])
    >>> features = torch.tensor([[0.81, 0.99], [0.69, 0.56], [0.83, 0.20], [0.59, 0.11]])
    >>> data_loader = DataLoader(
    >>>    dataset=TensorDataset(features, labels),
    >>>    batch_sampler=MLStratifiedBatchSampler(labels=labels, batch_size=2, random_state=1), shuffle=False)
    >>> print(next(iter(data_loader)))
    [tensor([[0.8100, 0.9900],
        [0.6900, 0.5600]]), tensor([[0, 1],
        [1, 0]])]
    """
    def __init__(self,
                 labels: torch.tensor,
                 batch_size: int,
                 limit_replacement: bool = False,
                 random_state=None):
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive non-zero integer.")
        self.labels = labels
        self.random_state = set_random_state(random_state)
        self.batch_size = batch_size
        self.n_batches = math.ceil(self.labels.shape[0] / self.batch_size)
        self.num_classes = self.labels.shape[1]

        # Chunks are the increment used within a batch to sample from the different pools - set so that each class is
        # guaranteed a minimum sample size of BatchSize//NumClasses.
        if self.batch_size < self.num_classes:
            raise ValueError("The batch_size is less than the number of labels - it is not possible to have every label"
                             "present.")
        chunk_size = self.batch_size // self.num_classes
        self._batch_chunks = [chunk_size for _ in range(self.batch_size // chunk_size)]
        self._batch_chunks[-1] += self.batch_size - np.sum(self._batch_chunks)

        # List of lists of example indices per class
        self.class_indices = get_sample_pool(self.labels.cpu().numpy())
        self.limit_replacement = limit_replacement
        if self.limit_replacement:
            self._class_indices_cache = self.class_indices.copy()

    def __iter__(self) -> Iterator[List[int]]:
        for batch_i in range(self.n_batches):
            # Construct batch using chunk size iteratively
            sample_sizes = np.zeros(self.num_classes)
            batch_idx = []
            for chunk_i, chunk_i_size in enumerate(self._batch_chunks):
                # Find group with the least sample size in batch - select random if tie
                smallest_group = self.random_state.choice(np.arange(self.num_classes)[sample_sizes == sample_sizes.min()])
                chunk_idx = self._sample(smallest_group, size=chunk_i_size)
                batch_idx += chunk_idx
                sample_sizes += np.nansum(self.labels[chunk_idx, :], axis=0)
            yield batch_idx

    def _sample(self, group: int, size: int) -> List[int]:
        if self.limit_replacement:
            return self._limited_replacement_sampling(group, size)
        else:
            return self._replacement_sampling(group, size)

    def _limited_replacement_sampling(self, group: int, size: int) -> List[int]:
        group_pool = self._class_indices_cache[group]
        if len(group_pool) == size:
            sample = group_pool
            self._class_indices_cache[group] = self.class_indices[group].copy()
        elif len(group_pool) < size:
            sample = group_pool
            group_pool = self.class_indices[group].copy()
            new_cycle_samples = self.random_state.choice(group_pool, size, replace=False).tolist()
            # Remove samples
            self._class_indices_cache[group] = np.setdiff1d(group_pool, new_cycle_samples).tolist()
            # Combine
            sample += new_cycle_samples
        else:
            sample = self.random_state.choice(group_pool, size, replace=False).tolist()
            self._class_indices_cache[group] = np.setdiff1d(group_pool, sample).tolist()
        return sample

    def _replacement_sampling(self, group: int, size: int) -> List[int]:
        group_pool = self.class_indices[group]
        samples = self.random_state.choice(group_pool, size, replace=True).tolist()
        return samples

    def __len__(self) -> int:
        return self.labels.shape[0]


# def evaluate_behavior(dataloader):
#     from torch_fairness.resampling import imbalance_ratio
#     batch_sizes = []
#     ir = []
#     number_of_batches_with_infinite = 0
#     number_of_batches = 0
#     sample_size = data_loader.dataset.__len__()
#     n_sampled = 0
#     for n_batches, batch in enumerate(dataloader):
#         # Calculate
#         features, sensitive = batch
#         sensitive = sensitive.numpy()
#         ir_i = imbalance_ratio(np.nansum(sensitive, axis=0))
#
#         # Update
#         batch_sizes.append(features.shape[0])
#         number_of_batches_with_infinite += np.isinf(ir_i).any()
#         mean_ir_i = np.mean(ir_i)
#         ir.append(mean_ir_i)
#         number_of_batches += 1
#         n_sampled += features.shape[0]
#     print(f"Number of batches {number_of_batches}")
#     print(f"Expected SS - {sample_size}, Received SS - {n_sampled}")
#     print(f"Batch size range: {np.min(batch_sizes)}-{np.max(batch_sizes)}")
#     print(f"Proportion unstable batches - {np.round(number_of_batches_with_infinite/number_of_batches, 2)}")
#     print(f"IR Mean - {np.mean(ir)}")
#     print(f"IR Var - {np.var(ir)}")
#
#
# if __name__ == '__main__':
#     from torch.utils.data import TensorDataset, DataLoader
#     from torch_fairness.data import SensitiveTransformer
#
#     np.random.seed(0)
#     torch.manual_seed(0)
#     n_random_samples = 20
#     features = torch.arange(n_random_samples)[:, None].to(torch.float32)
#     sensitive = pd.DataFrame([np.random.choice([0, 1, 2], n_random_samples, p=[0.5, 0.3, 0.2]),np.random.choice([0, 1], n_random_samples, p=[0.7, 0.3])]).T
#     sensitive.columns = ['race', 'gender']
#     transformer = SensitiveTransformer(minimum_sample_size=1)
#     sensitive = transformer.fit_transform(sensitive)
#     sensitive_map = transformer.sensitive_map
#     batch_size = 8
#     n_batches = int(sensitive.shape[0]/batch_size)
#
#     # Versions 1 - Still unstable (~5%) and there is a range of batch size
#     # data_loader = DataLoader(
#     #     dataset=TensorDataset(features, torch.tensor(sensitive)),
#     #     batch_sampler=StratifiedBatchSampler(torch.tensor(sensitive), batch_size=batch_size)
#     # )
#     # evaluate_behavior(data_loader)
#
#     # NEW
#     data_loader = DataLoader(
#         dataset=TensorDataset(features, torch.tensor(sensitive)),
#         batch_sampler=MLStratifiedBatchSampler(labels=torch.tensor(sensitive), batch_size=batch_size, random_state=1, limit_replacement=False)
#     )
#     out_idx = []
#     for n_batches, batch in enumerate(data_loader):
#         out_idx += batch[0].squeeze().to(int).numpy().tolist()
#     print(f"Number of duplicates -  {len(out_idx) - len(set(out_idx))}")
#     evaluate_behavior(data_loader)
