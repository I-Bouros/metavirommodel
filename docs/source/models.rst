***************************************
List of Rodent Diseases Dynamics Models
***************************************

This section documents all epidemiological models of rodent disease dynamics included in this package.
All models follow a Gillespie SSA approach, in which all indvidual rodent are independently tracked
(based on their current infection status), as well as each individual predator, when Lotka-Volterra dynamics are included.
No waning immunity is accounted for in any of the models considered.

The first class of models only follow rodent infection and population dynamics, for a range of possible options for the birth term.
The default model assumes a constant birth rate, subject to temporal variations due to seasonality or precipitation levels.
The second and third models replace this birth term with an exponential, respectively, a logistic growth term, for more realsitic population dynamics.

The other class of models extend the first ones, by introducing a predator species to induce cyclicity in the overall rodent population size. Both natural
death and death due to predations are independently considered for all models included in this class of Lotka-Volterra-type models.

.. currentmodule:: metavirommodel

Overview:
  - Stochastic SIR model
    - :class:`Metaviromodel`
    - :class:`ExponentialGrowthMetaviromodel`
    - :class:`LogisticGrowthMetaviromodel`
    - :class:`constant_func`

  - Stochastic SIR model with Lotka-Volterra dynamics
    - :class:`PreyPredMetaviromodel`
    - :class:`constant_func`

Non-predator Stochastic SIR Models
**********************************
Constant Birth Stochastic SIR model
***********************************

.. autoclass:: Metaviromodel
  :members:

Exponential Birth Stochastic SIR model
***********************************

.. autoclass:: ExponentialGrowthMetaviromodel
  :members:

Logistic Birth Stochastic SIR model
***********************************

.. autoclass:: LogisticGrowthMetaviromodel
  :members:

Predator Stochastic SIR Models
******************************
Logistic Birth Stochastic SIR model with predator dynamics
**********************************************************

.. autoclass:: PreyPredMetaviromodel
  :members: