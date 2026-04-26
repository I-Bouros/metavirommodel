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

.. autoclass:: MVRCtValInfer
  :members:

.. autoclass:: MVRCtValLogLik
  :members:

.. autoclass:: MVRCtValLogPrior
  :members:

Logistic Birth Rate Stochastic SIR Model
****************************************

.. autoclass:: LogisticGrowthMVRCtValInfer
  :members:

.. autoclass:: LogisticGrowthMVRCtValLogLik
  :members:

.. autoclass:: LogisticMVRCtValLogPrior
  :members:

Exponential Birth Rate Stochastic SIR Model
*******************************************

.. autoclass:: ExponentialGrowthMVRCtValInfer
  :members:

.. autoclass:: ExponentialGrowthMVRCtValLogLik
  :members:

.. autoclass:: ExponentialGrowthMVRCtValLogPrior
  :members:

Viral Read Data-informed Inference Classes
******************************************

This section includes all inference classes which use Viral read-based metaviromic 
data for model parameter inference, organised by the population model.

Constant Birth Rate Stochastic SIR Model
****************************************

.. autoclass:: MVRVirReadInfer
  :members:

.. autoclass:: MVRVirReadLogPrior
  :members:

.. autoclass:: MVRVirReadLogLik
  :members:

Logistic Birth Rate Stochastic SIR Model
****************************************

.. autoclass:: LogisticGrowthMVRVirReadInfer
  :members:

.. autoclass:: LogisticGrowthMVRVirReadLogLik
  :members:

Exponential Birth Rate Stochastic SIR Model
*******************************************

.. autoclass:: ExponentialGrowthMVRVirReadInfer
  :members:

.. autoclass:: ExponentialGrowthMVRVirReadLogLik
  :members:

Logistic Birth Rate Stochastic SIR Model with Lotka-Volterra dynamics
*********************************************************************

.. autoclass:: PreyPredGrowthMVRVirReadInfer
  :members:

.. autoclass:: PreyPredGrowthMVRVirReadLogLik
  :members:
