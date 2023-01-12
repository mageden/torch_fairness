
Resampling Methods
==================

Sensitive attributes often display class imbalance between the multiple labels. While ideally data collection involves a
balanced sample across attributes, in reality this is often not possible. Resampling sensitive attributes to reduce the
class imbalance can improve model training. Since fairness attributes are multi-label with often correlated labels,
resampling methods must be able to account for label overlap (unlike traditional multi-class resamplers).

.. currentmodule:: torch_fairness.resampling

.. autosummary::
   :toctree: generated/

   MLROS
   MLRUS
   MLSMOTE
   MLeNN


Utility Functions
-----------------

These are measures of imbalance used for creating resampled datasets.

.. currentmodule:: torch_fairness.resampling

.. autosummary::
   :toctree: generated/

   imbalance_ratio
   scumble
   adjusted_hamming_distance