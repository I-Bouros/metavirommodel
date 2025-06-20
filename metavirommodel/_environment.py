#
# This file is part of metavirommodel
# (https://github.com/I-Bouros/metavirommodel)
# which is released under the BSD 3-clause license. See accompanying LICENSE.md
# for copyright notice and full license details.
#

import numpy as np
import pandas as pd
import math


class Environment(object):
    r"""Environment Class:
    Base class for creating callable functions for the time-dependent rate of
    birth of susceptible individuals, the rate of death of the susceptibles and
    recovered and the rate of death of the infectious individuals.

    These functions take as argument the time of year at which the birth and
    death rates are evaluated and return a value depending on the amount of
    precipitation in the time prior to the evaluation time.

    Methods
    -------
    _set_rate_rule: computes the value of the birth or death rates based on a
                    rainfall criterion.

    """
    def __init__(self):
        super(Environment, self).__init__()

    def _set_rate_rule(self):
        """
        Computes the value of the birth or death rates based on a rainfall
        criterion.

        """

    def __call__(self, t_cal):
        """
        Returns the value of the birth or death rate at the specified time of
        the year.

        Parameters
        ----------
        t_cal
            (int) current time according to the calendar date.
        """
        raise NotImplementedError


class BirthRatePrec(Environment):
    r"""

    Parameters
    ----------
    precipitation_data
        (pandas Dataframe) dataframe of the amounts of precipitation and
            their corresponding calendar date.
    parameters:
        (list) list of the hyper-parameters underpining the method for
        computing the birth rate: the maximum birth rate (bM), the sensitivity
        to precipitation parameter (rho), and, if present, the prescribed time
        window for computing the average previous precipitation levels used to
        compute the birth rate.

    """
    def __init__(self, precipitation_data, parameters):
        super(BirthRatePrec, self).__init__()

        # Set the maximum birth rate and sensitivity to precipitation
        self._check_parameters(parameters)
        self.bM = parameters[0]
        self.rho = parameters[1]

        # Set precipitation data and compute average precipitation
        self._check_precipitation_data(precipitation_data)

        self._precipitation_data = precipitation_data
        if len(parameters) < 3:
            self.process_precipitation_data()
        else:
            if not isinstance(parameters[2], int):
                raise TypeError('Time range must be integer.')
            self.process_precipitation_data(parameters[2])

        # Set the rule after which the value of the birth rate is calculated
        self._set_rate_rule()

    def process_precipitation_data(self, time_range=30):
        """
        Process the precipitation data used to compute the current value of the
        birth rate of susceptibles.

        Parameters
        ----------
        time_range
            (int or float) the length of the time interval which to compute
            the average levels of rainfall.

        """
        average_precipitation = []

        # Pad with zeros the time points where we have no information on
        # the precipitation
        data_times = self._precipitation_data['Day']

        padded_prec_data = self._precipitation_data.set_index('Day').reindex(
            range(
                min(data_times), max(data_times)+1)
                ).fillna(0).reset_index()

        prec_data = padded_prec_data['Precmm'].to_numpy()

        # Compute the average rainfall over the last time_range days
        for d in padded_prec_data['Day'].values:
            if d < time_range:
                average_precipitation.append(
                    np.sum(prec_data[:d])/time_range)
            else:
                average_precipitation.append(
                    np.mean(prec_data[(d-time_range):d]))

        self.avg_prec = average_precipitation

    def _check_precipitation_data(self, precipitation_data):
        """
        Checks that the precipitation data has the correct format.

        """

        if not issubclass(type(precipitation_data), pd.DataFrame):
            raise TypeError('Precipitation data has to be a dataframe.')

        if 'Day' not in precipitation_data.columns:
            raise ValueError(
                'No time column with this name in given'
                'precipitation data.')

        if 'Precmm' not in precipitation_data.columns:
            raise ValueError(
                'No precipitation column with this name in given'
                'precipitation data.')

    def _check_parameters(self, parameters):
        """
        Checks that the hyper-parameters have the correct format.

        """
        if not isinstance(parameters, list):
            raise TypeError('Parameters must be given in a list format.')
        if len(parameters) != 3 and len(parameters) != 2:
            raise ValueError(
                'List of parameters needs to be of length 2 or 3.')
        for _ in parameters:
            if not isinstance(_, (int, float)):
                raise TypeError(
                    'All parameters must be integer or float.')
            if _ < 0:
                raise ValueError('All parameters must be => 0.')

    def _set_rate_rule(self):
        r"""
        Computes the value of the birth rates based on a rainfall criterion.

        .. math::
            b(t) = |b_0 \sin(2 pi (\frac{\lfloor t/7 \rfloor \}{52} - \omega))
            | + b_0 \sin(2 pi (\frac{\lfloor t/7 \rfloor \}{52} - \omega))

        where :math:`b_M` is the baseline birth rate and :math:`\rho` is the
        birth rate phase used to compute the birth rate.

        """
        self.birth_rate = lambda t: self.bM * self.avg_prec[
            np.floor(t).astype(int)-1] / (self.rho + self.avg_prec[
                np.floor(t).astype(int)-1])

    def __call__(self, t_cal):
        """
        Returns the value of the birth rate at the specified time of the year.

        Parameters
        ----------
        t_cal
            (int) current time according to the calendar date.
        """
        if t_cal > len(self.avg_prec):
            raise ValueError(
                'Time of simulation outside precipitation data bounds.')

        return self.birth_rate(t_cal)


class BirthRateSeason(Environment):
    r"""

    Parameters
    ----------
    parameters:
        (list) list of the hyper-parameters underpining the method for
        computing the birth rate: the baseline birth rate (b0) and the birth
        rate phase used to compute the birth rate (w).

    """
    def __init__(self, parameters):
        super(BirthRateSeason, self).__init__()

        # Set the baseline birth rate and birth rate phase
        self._check_parameters(parameters)
        self.b0 = parameters[0]
        self.w = parameters[1]

        # Set the rule after which the value of the birth rate is calculated
        self._set_rate_rule()

    def _check_parameters(self, parameters):
        """
        Checks that the hyper-parameters have the correct format.

        """
        if not isinstance(parameters, list):
            raise TypeError('Parameters must be given in a list format.')
        if len(parameters) != 2:
            raise ValueError(
                'List of parameters needs to be of length 2.')
        for _ in parameters:
            if not isinstance(_, (int, float)):
                raise TypeError(
                    'All parameters must be integer or float.')
            if _ < 0:
                raise ValueError('All parameters must be => 0.')

    def _set_rate_rule(self):
        r"""
        Computes the value of the birth rates based on a seasonality criterion.

        .. math::
            b(t) = |b_0 \sin(2 pi (\frac{\lfloor t/7 \rfloor \}{52} - \omega))
            | + b_0 \sin(2 pi (\frac{\lfloor t/7 \rfloor \}{52} - \omega))

        where :math:`b_0` is the baseline birth rate and :math:`\omega` is the
        birth rate phase used to compute the birth rate.

        """

        self.birth_rate = lambda t: np.abs(
            self.b0 * np.sin(2*math.pi*(np.floor(t/7)/52 - self.w))) + \
            self.b0 * np.sin(2*math.pi*(np.floor(t/7)/52 - self.w))

    def __call__(self, t_cal):
        """
        Returns the value of the birth rate at the specified time of the year.

        Parameters
        ----------
        t_cal
            (int) current time according to the calendar date.
        """
        return self.birth_rate(t_cal)
