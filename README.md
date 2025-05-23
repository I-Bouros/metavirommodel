# Renewal Models for vector-borne diseases

[![Run Unit Tests on multiple OS](https://github.com/SABS-R3-Epidemiology/metavirommodel/actions/workflows/os-unittests.yml/badge.svg)](https://github.com/SABS-R3-Epidemiology/metavirommodel/actions/workflows/os-unittests.yml)
[![Run Unit Tests on multiple python versions](https://github.com/SABS-R3-Epidemiology/metavirommodel/actions/workflows/python-version-unittests.yml/badge.svg)](https://github.com/SABS-R3-Epidemiology/metavirommodel/actions/workflows/python-version-unittests.yml)
[![Documentation Status](https://readthedocs.org/projects/metavirommodel/badge/?version=latest)](https://metavirommodel.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/SABS-R3-Epidemiology/metavirommodel/branch/main/graph/badge.svg?token=UBJG0AICF9)](https://codecov.io/gh/SABS-R3-Epidemiology/metavirommodel/)
[![DOI](https://zenodo.org/badge/305988905.svg)](https://doi.org/10.5281/zenodo.14166376)

In this package, we use branching processes to model the time-dependent reproduction number (the number of cases each infected individual will subsequently cause) of an infectious disease.

All features of our software are described in detail in our
[full API documentation](https://metavirommodel.readthedocs.io/en/latest/).

A web app for performing inference for branching process models is included in this package. Instructions for accessing the app are available [here](https://sabs-r3-epidemiology.github.io/metavirommodel/).

More details on branching process models and inference can be found in these
papers:

## References

[1]
R. Creswell,<sup>†</sup> D. Augustin,<sup>†</sup> I. Bouros,<sup>†</sup> H. J. Farm,<sup>†</sup> S. Miao,<sup>†</sup> A. Ahern,<sup>†</sup> M. Robinson, A. Lemenuel-Diot, D. J. Gavaghan, B. C. Lambert and R. N. Thompson: “Heterogeneity in the onwards transmission risk between local and imported cases affects practical estimates of the time-dependent reproduction number,” <em>Phil. Trans. R. Soc. A.</em> 380: 20210308 (2022).

[2]
Cori A, Ferguson NM, Fraser C, Cauchemez S. (2013). A new framework and
software to estimate time-varying reproduction numbers during epidemics.
American Journal of Epidemiology 178(9): 1505-12.

[3]
Thompson RN, Stockwin JE, van Gaalen RD, Polonsky JA, Kamvar ZN, Demarsh PA,
Dahlqwist E, Li S, Miguel E, Jombart T, Lessler J. (2019). Improved inference of
time-varying reproduction numbers during infectious disease outbreaks.
Epidemics 29: 100356.

## Installation procedure

***
One way to install the module is to download the repositiory to your machine of choice and type the following commands in the terminal.

```bash
git clone https://github.com/SABS-R3-Epidemiology/metavirommodel.git
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

# create a simple branching process model with prescribed initial R and serial interval
metavirommodel.metavirommodelModel(initial_r=0.5, serial_interval=[0, 0.15, 0.52, 0.3, 0.01])

# create branching process model with local and imported cases with prescribed initial R
# and serial interval
# set imported cases data
libr_model_1 = metavirommodel.LocImpmetavirommodelModel(
  initial_r=2, serial_interval=np.array([1, 2, 3, 2, 1]), epsilon=1)
libr_model_1.set_imported_cases(times=[1, 2.0, 4, 8], cases=[5, 10, 9, 2])

# create the posterior of a branching process model for multiple daily serial intervals
# and incidence data contained in the dataframe df; prior distribution is Gamma with
# parameters alpha and beta (shape, rate)
metavirommodel.metavirommodelPosteriorMultSI(
  inc_data=df, daily_serial_intervals=[[1, 2], [0, 1]], alpha=1, beta=0.2)
```

More examples on how to use the classes and features included in this repository can be found [here](https://github.com/SABS-R3-Epidemiology/metavirommodel/tree/main/examples).

## Multiple group population models

In their most basic form, the branching processes modelling approach assume that all previous infections occurring on the same day contribute in equal measure to the present incidence of infection. However, this assumption is not generally true for most epidemic scenarios, where different population groups share different epidemic burdens. 

Therefore, we have now extendend the metavirommodel package to offer users the possibility to run both running forward simulation and perform Rt inference for both the overall and group-specific reproduction numbers using a multiple-group population branching process. The approach implemented bypasses the need to use the next-generation matrix approach, as detailed in our [preprint]().

To recreate our analyses for the suitability of multiple group renewal equations and Rt inference, please rerun the notebooks found [here](https://github.com/SABS-R3-Epidemiology/metavirommodel/tree/main/metavirommodel/results/heterogeneity).

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[BSD-3-Clause](https://opensource.org/licenses/BSD-3-Clause)
