
Batch Samplers
==============

Fair machine learning models add an additional challenge during batching - stratification along the multi-label
sensitive attributes. A naive approach of randomly sampling batches may work with large batch sizes and a small number
of sensitive attributes. However, in real world settings with multiple sensitive attributes, this is likely to result in
unstable batches where sensitive attributes are not represented in the batch.

Multi-label stratification improves the reliability of the batches during training, but is not as straightforward as
multi-class stratification due to labels often not being independent in the sample (e.g., race and gender).

.. currentmodule:: torch_fairness.sampler

.. autosummary::
   :toctree: generated/

   MLStratifiedBatchSampler
