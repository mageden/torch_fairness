

Fairness Metrics
================

Group fairness concepts are typically centered around three aspects: independence, separation, and sufficiency.
Fairness measures are organized underneath these larger conceptual frameworks and relate to a specific instantiation
(e.g., demographic parity relates to independence).

Independence
------------

Independence corresponds to the goal of making the score of a model/decision (:math:`\hat{Y}`) independent of an
individual's sensitive attributes (:math:`S`).

.. math::
    S \perp \hat{Y}

.. currentmodule:: torch_fairness.metrics

.. autosummary::
   :toctree: generated/

   DemographicParity
   SmoothedEmpiricalDifferentialFairness
   WeightedSumofLogs
   MeanDifferences
   MMDFairness
   AbsoluteCorrelation

Separation
----------

Separation corresponds to the goal of making the score of a model/decision (:math:`\hat{Y}`) independent of an
individual's sensitive attributes (:math:`S`) conditional on the label (:math:`Y`).

.. math::
    S \perp \hat{Y} | Y

In otherwords, it is a relaxed form of independence where group differences are allowed as long as they are due to true
differences in the underlying distribution of labels.

.. currentmodule:: torch_fairness.metrics

.. autosummary::
   :toctree: generated/

   FalsePositiveRateBalance
   EqualOpportunity
   EqualizedOdds
   BalancedPositive
   BalancedNegative
   MinimaxFairness

Sufficiency
-----------

Sufficiency corresponds to the goal of making the label (:math:`Y`) independent of an
individual's sensitive attributes (:math:`S`) conditional on the score of a model/decision (:math:`\hat{Y}`)

.. math::
    S \perp Y | \hat{Y}

.. currentmodule:: torch_fairness.metrics

.. autosummary::
   :toctree: generated/

   PredictiveParity
   ConditionalUseAccuracyParity


Other
-----

While independence, sufficiency, and seperation encompass most fairness definitions, some do not neatly fall into any of
them alone.

.. currentmodule:: torch_fairness.metrics

.. autosummary::
   :toctree: generated/

   CrossPairDistance
   AccuracyEquality
