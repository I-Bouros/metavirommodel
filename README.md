# Metavirommodel: Metaviromic-informed epidemiological models for rodent infectious

[![Run Unit Tests on multiple OS](https://github.com/I-Bouros/metavirommodel/actions/workflows/os-unittests.yml/badge.svg)](https://github.com/I-Bouros/metavirommodel/actions/workflows/os-unittests.yml)
[![Run Unit Tests on multiple python versions](https://github.com/I-Bouros/metavirommodel/actions/workflows/python-version-unittests.yml/badge.svg)](https://github.com/I-Bouros/metavirommodel/actions/workflows/python-version-unittests.yml)
[![Documentation status](https://github.com/I-Bouros/metavirommodel/actions/workflows/doctest.yml/badge.svg)](https://github.com/I-Bouros/metavirommodel/actions/workflows/doctest.yml)
[![codecov](https://codecov.io/gh/I-Bouros/metavirommodel/branch/main/graph/badge.svg?token=UBJG0AICF9)](https://codecov.io/gh/I-Bouros/metavirommodel/)

In this package, we use metaviromic data, in the form of both Ct values and viral read load counts to the infer infectious disease dynamics in wildlife populations of rodents for a multitude of environmental and ecological contexts, such as:
- constant, logistic and exponential birth rates;
- precipitiation-  or season-dependent birth rates of susceptible rodents [1] [2];
- multiple-species ecological models, when an additional predator species is also modelled, using Lotka-Volterra population dynamics [3].

All features of our software are described in detail in our
[full API documentation](https://metavirommodel.readthedocs.io/en/latest/). 

More details on metaviromic-informed epidemic models and inference can be found in these papers:

## References

[1]
Nuismer SL, Remien CH, Basinski AJ, Varrelman T, Layman N, Rosenke K, et al. _Bayesian estimation of Lassa virus epidemiological parameters: Implications for spillover prevention using wildlife vaccination_. PLoS Negl Trop Dis
14(9): **e0007920** (2020). DOI:10.1371/journal.pntd.0007920

[2]
Diana Erazo et al., _Who acquires infection from whom? Estimating herpesvirus transmission rates between wild rodent host groups_. Epidemics35(2021). DOI:10.1016/j.epidem.2021.100451

[3]
Hanski, I., E. Korpima¨ki., _Microtine rodent dynamics in northern Europe: parameterized models for the predator-prey interaction_. Ecology 76:840–850 (1995). DOI:10.2307/1939349

[4]
James A. Hay et al., _Estimating epidemiologic dynamics from cross-sectional viral load distributions_. Science373,**eabh0635(2021)**. DOI:10.1126/science.abh0635

## Installation procedure

***
One way to install the module is to download the repositiory to your machine of choice and type the following commands in the terminal.

```bash
git clone https://github.com/I-Bouros/metavirommodel.git
cd ../path/to/the/file
```

A different method to install this is using `pip`:

```bash
pip install -e .
```

## Usage

```python
import metavirommodel
import numpy as np

# create a simple stochastic SIR compartmental model with precipitation-dependent growth rate
# run forward simulation with prescribed rates and initial population compartment sizes
algorithm = metavirommodel.Metaviromodel()

precipitation_data = pd.read_csv(os.path.join('../../data/precipitation/Precipitation.csv'))
theta = metavirommodel.BirthRatePrec(precipitation_data, parameters=[0.7, 2.8, 30])

algorithm.simulate_fixed_times(parameters=[380, 20, 0, theta, 0, 0, 0.2, 0.66], start_time=1, end_time=300)

# create a simple stochastic SIR compartmental model with logistic growth rate
# run forward simulation with prescribed rates and initial population compartment sizes
logistic_algorithm = mm.LogisticGrowthMetaviromodel(carrying_capacity=400)
logistic_algorithm.simulate_fixed_times(parameters=[380, 20, 0, 0.05, 0, 0, 0.2, 0.66], start_time=1, end_time=300)

# create the posterior controller class of a stochastic SIR compartmental model for viral read count data contained in the dataframe df and for prescribed generation time distribution in the list generation_times;
# and run a sampling algorithm inference routine
posterior = metavirommodel.inference.MVRVirReadInfer(model=model, generation_times= generation_times)

posterior.read_viral_read_data(df, parameters_vl=[3, 7, 5, 15, np.inf, 2, 3880, 480, 2, 0.4, 0.25])
posterior.inference_problem_setup(num_iter=1000)
```

To recreate our analyses for the suitability of multiple group renewal equations and Rt inference, please rerun the notebooks found [here](https://github.com/I-Bouros/metavirommodel/tree/main/metavirommodel/results).

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[BSD-3-Clause](https://opensource.org/licenses/BSD-3-Clause)
