*****************
Inference Classes
*****************

This section documents the classes used for the parameter inference of epidemiological models
curated in the metavirommodel package. Different inference controller, log-likelihood and prior
classes are considered for each model type and data source (Ct Values versus Viral read data).

.. currentmodule:: metavirommodel.inference

Overview:

- Inference & Optimisation Controller Classes:
    - :class:`MVRCtValInfer`
    - :class:`LogisticGrowthMVRCtValInfer`
    - :class:`ExponentialGrowthMVRCtValInfer`
    - :class:`MVRVirReadInfer`
    - :class:`LogisticGrowthMVRVirReadInfer`
    - :class:`ExponentialGrowthMVRVirReadInfer`
    - :class:`PreyPredGrowthMVRVirReadInfer`

- Log-likelihood Classes:
    - :class:`MVRCtValLogLik`
    - :class:`LogisticGrowthMVRCtValLogLik`
    - :class:`ExponentialGrowthMVRCtValLogLik`
    - :class:`MVRVirReadLogLik`
    - :class:`LogisticGrowthMVRVirReadLogLik`
    - :class:`ExponentialGrowthMVRVirReadLogLik`
    - :class:`PreyPredGrowthMVRVirReadLogLik`

- Prior Classes:
    - :class:`MVRCtValLogPrior`
    - :class:`LogisticMVRCtValLogPrior`
    - :class:`ExponentialMVRCtValLogPrior`


Ct Value Data-informed Inference Classes
****************************************

This section includes all inference classes which use Ct value-based metaviromic
data for model parameter inference, organised by the population model.

Constant Birth Rate Stochastic SIR Model
****************************************

.. autoclass:: inference_ct_values.MVRCtValInfer
  :members:

.. autoclass:: inference_ct_values.MVRCtValLogLik
  :members:

.. autoclass:: inference_ct_values.MVRCtValLogPrior
  :members:

Logistic Birth Rate Stochastic SIR Model
****************************************

.. autoclass:: inference_ct_values.LogisticGrowthMVRCtValInfer
  :members:

.. autoclass:: inference_ct_values.LogisticGrowthMVRCtValLogLik
  :members:

.. autoclass:: inference_ct_values.LogisticMVRCtValLogPrior
  :members:

Exponential Birth Rate Stochastic SIR Model
*******************************************

.. autoclass:: inference_ct_values.ExponentialGrowthMVRCtValInfer
  :members:

.. autoclass:: inference_ct_values.ExponentialGrowthMVRCtValLogLik
  :members:

.. autoclass:: inference_ct_values.ExponentialGrowthMVRCtValLogPrior
  :members:

Viral Read Data-informed Inference Classes
******************************************

This section includes all inference classes which use Viral read-based metaviromic 
data for model parameter inference, organised by the population model.

Constant Birth Rate Stochastic SIR Model
****************************************

.. autoclass:: inference_viral_reads.MVRVirReadInfer
  :members:

.. autoclass:: inference_viral_reads.MVRVirReadLogPrior
  :members:

.. autoclass:: inference_viral_reads.MVRVirReadLogLik
  :members:

Logistic Birth Rate Stochastic SIR Model
****************************************

.. autoclass:: inference_viral_reads.LogisticGrowthMVRVirReadInfer
  :members:

.. autoclass:: inference_viral_reads.LogisticGrowthMVRVirReadLogLik
  :members:

Exponential Birth Rate Stochastic SIR Model
*******************************************

.. autoclass:: inference_viral_reads.ExponentialGrowthMVRVirReadInfer
  :members:

.. autoclass:: inference_viral_reads.ExponentialGrowthMVRVirReadLogLik
  :members:

Logistic Birth Rate Stochastic SIR Model with Lotka-Volterra dynamics
*********************************************************************

.. autoclass:: inference_viral_reads.PreyPredGrowthMVRVirReadInfer
  :members:

.. autoclass:: inference_viral_reads.PreyPredGrowthMVRVirReadLogLik
  :members:
