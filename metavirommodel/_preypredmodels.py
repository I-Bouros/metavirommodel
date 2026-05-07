#
# PreyPredMetaviromodel Class
#
# This file is part of metavirommodel
# (https://github.com/I-Bouros/metavirommodel)
# which is released under the BSD 3-clause license. See accompanying LICENSE.md
# for copyright notice and full license details.
#
"""
This script contains code for modelling the between-host dynamics of viral
transmission in rodents using an SIR agent-based modellling framework when the
predator population is also accounted for.

"""

import metavirommodel as mvr
import numpy as np
from scipy.stats import uniform
import math


class PreyPredMetaviromodel(mvr.LogisticGrowthMetaviromodel):
    r"""PreyPredMetaviromodel Class:
    Base class for the forward simulation of the epidemic transmission dynamic
    of a population of rodents when a logistic population growth and both prey
    (rodent) and predator population dynamics are assumed.

    Four types of individuals are considered based on their serological status
    - susceptible individuals (S), infectious (I), recovered (R) - all of
    which are rodent / prey; and finally a compartment for the predators (P).

    Susceptible individuals are born according to a logistic growth model and
    die at the same rate as those recovered. A different death rate due to
    disease is considered for the infected individuals. A susceptible
    individuals goes on to become infected at a constant rate.

    The system of equations that describe the isolated possible events that can
    occur in the rodent population are

    .. math::
        :nowrap:

        \begin{eqnarray}
            S + I  &\xrightarrow{\beta} 2 * I \\
            I  &\xrightarrow{\gamma} R \\
            \emptyset  &\xrightarrow{rN(1-\frac{N}{K})} S \\
            S &\xrightarrow{\mu} \emptyset \\
            I &\xrightarrow{\nu} \emptyset \\
            R &\xrightarrow{\mu} \emptyset \\
            S + P &\xrightarrow{\frac{\alpha}{N+D}} P \\
            I + P &\xrightarrow{\frac{\alpha'}{N+D}} P \\
            R + P &\xrightarrow{\frac{\alpha}{N+D}} P
        \end{eqnarray}

    where :math:`\mu` and :math:`\nu` are the rates of natural death in
    the susceptibles and recovered, and infectious respectively,
    :math:`r` is the growth rate in the susceptibles and :math:`K` is the
    rodent carrying capacity. :math:`\beta` is the transmission rate and
    :math:`\gamma` is the recovery rate.

    """
    def __init__(self, carrying_capacity, pred_carrying_capacity,
                 prey_carrying_capacity, N_crit, delay):
        super(mvr.Metaviromodel, self).__init__()

        if isinstance(carrying_capacity, (float, int)):
            # Same birth rate for every day of the year
            self._carry_cap = mvr.constant_func(carrying_capacity)
        else:
            self._carry_cap = carrying_capacity

        if isinstance(pred_carrying_capacity, (float, int)):
            # Same birth rate for every day of the year
            self.d = mvr.constant_func(pred_carrying_capacity)
        else:
            self.d = pred_carrying_capacity

        if isinstance(prey_carrying_capacity, (float, int)):
            # Same birth rate for every day of the year
            self.q = mvr.constant_func(prey_carrying_capacity)
        else:
            self.q = prey_carrying_capacity

        if not isinstance(delay, (float, int)):
            raise TypeError('Parameters must be given in a list format.')
        self.rho = delay

        if not isinstance(N_crit, (float, int)):
            raise TypeError('Parameters must be given in a list format.')
        self.N_crit = N_crit

        self._output_names = ['S', 'I', 'R', 'P']
        self._parameter_names = [
            'S0', 'I0', 'R0', 'P0', 'theta', 'mu', 'nu', 'beta', 'gamma',
            'pred_birth', 'pred_death_H', 'pred_death_L']

        # The default number of outputs is 4,
        # i.e. S, I and R, and P
        self._n_outputs = len(self._output_names)
        # The default number of outputs is 12,
        # i.e. 4 initial conditions and 8 parameters
        self._n_parameters = len(self._parameter_names)

        self._output_indices = np.arange(self._n_outputs)

    def one_step_gillespie(self, t_cal, i_S, i_I, i_R, i_P):
        """
        Computes one step in the Gillespie algorithm to determine the
        counts of the different types of individuals present in the population
        at present. Returns time to next reaction and the tuple state of the
        system, as well as the type of reaction that occured.

        Parameters
        ----------
        t_cal
            (int) current time according to the calendar date.
        i_S
            (int) number of susceptibles (S) in the population at current time
            point.
        i_I
            (int) number of infectious individuals (I) in the population at
            current time point.
        i_R
            (int) number of recovered individuals (R) in the population at
            current time point.
        i_P
            (int) number of predators (P) in the population at current time
            point.

        """
        # Generate random number for reaction and time to next reaction
        u, u1 = uniform.rvs(size=2)

        self.N = sum((i_S, i_I, i_R))

        new_susc = 0
        new_infec = 0
        new_rec = 0

        # Compute propensities
        if self.N > 0:
            propens_1 = self.beta * i_S * i_I / self.N
            propens_2 = self.gamma * i_I
            propens_3 = self._compute_pred_birth(t_cal, i_P)
            propens_4 = self._compute_pred_death(t_cal, i_P)
            propens_5 = self._logistic_growth(
                t_cal, self._compute_theta(t_cal),
                i_S + i_I + i_R, self._carry_cap)
            propens_6 = self._compute_predation_S(t_cal, i_P) * i_S
            propens_7 = self._compute_predation_I(t_cal, i_P) * i_I
            propens_8 = self._compute_predation_S(t_cal, i_P) * i_R

            propens = np.array([
                propens_1, propens_2, propens_3, propens_4,
                propens_5, propens_6, propens_7, propens_8])
            sum_propens = np.empty(propens.shape)

            for e in range(propens.shape[0]):
                sum_propens[e] = np.sum(propens[:(e+1)]) / np.sum(propens)
            # Time to next reaction
            tau = np.log(1/u1) / np.sum(propens)

            if u < sum_propens[0]:
                # Susceptible becomes infected
                i_S += -1
                i_I += 1
                new_susc = -1
                new_infec = 1
            elif (u >= sum_propens[0]) and (u < sum_propens[1]):
                # Infected becomes recovered
                i_I += -1
                i_R += 1
                new_infec = -1
                new_rec = 1
            elif (u >= sum_propens[1]) and (u < sum_propens[2]):
                # New Predator
                i_P += 1
            elif (u >= sum_propens[2]) and (u < sum_propens[3]):
                # Predator dies
                i_P += -1
            elif (u >= sum_propens[3]) and (u < sum_propens[4]):
                # New susceptible
                i_S += 1
                new_susc = 1
            elif (u >= sum_propens[4]) and (u < sum_propens[5]):
                # Susceptible dies
                i_S += -1
                new_susc = -1
            elif (u >= sum_propens[5]) and (u < sum_propens[6]):
                # Infected dies
                i_I += -1
                new_infec = -1
            else:
                # Recovered dies
                i_R += -1
                new_rec = -1

        return (tau, i_S, i_I, i_R, new_susc, new_infec, new_rec, i_P)

    def gillespie_algorithm_fixed_times(self, start_time, end_time):
        """
        Runs the Gillespie algorithm for the population epidemic dynamics
        for the given times.

        Parameters
        ----------
        start_time
            (int) Time from which we start the simulation of the tumor.
        end_time
            (int) Time at which we end the simulation of the tumor.

        """
        # Create timeline vector
        times = np.arange(start_time, end_time+0.5, 1, dtype=np.int64)
        interval = end_time - start_time + 1

        # Split compartments into their types
        i_S, i_I, i_R, i_P = self.init_cond

        large_solution = []
        time_solution = []

        susc_history = []
        infect_history = []
        recov_history = []

        infect_times_history = []
        recov_infect_times_history = []

        solution = np.empty((interval, 4), dtype=np.int64)
        S_history = []
        I_history = []
        R_history = []

        I_times_history = []
        R_times_history = []

        current_time = start_time
        new_susc = 0
        new_infec = 0
        new_rec = 0
        self.last_used_id = i_S + i_I + i_R + 1
        infec_incidence = []

        while current_time <= end_time:
            time_solution.append(float(current_time))
            large_solution.append([i_S, i_I, i_R, i_P])

            if len(infect_history) > 0:
                # If an infection disappears
                if new_infec == -1:
                    infec_incidence.append(infec_incidence[-1])
                    # Read in the last structure of infections
                    current_susceptibles = susc_history[-1]
                    current_infections = infect_history[-1]
                    current_recovered = recov_history[-1]

                    current_infec_times = infect_times_history[-1]
                    current_recov_infec_times = recov_infect_times_history[-1]

                    # Select infection to disappear using a multinomial
                    # distribution
                    weights = current_time - np.asarray(current_infec_times)
                    if np.sum(weights) == 0:
                        elim_infec = np.random.choice(
                            range(len(current_infections)))
                    else:
                        elim_infec = np.random.choice(
                            range(len(current_infections)),
                            p=weights/np.sum(weights)
                            )

                    # Eliminate infection
                    new_current_infections = current_infections[
                        :(elim_infec)] + current_infections[(elim_infec+1):]
                    new_current_infec_times = current_infec_times[
                        :elim_infec] + current_infec_times[(elim_infec+1):]

                    susc_history.append(current_susceptibles)
                    infect_history.append(new_current_infections)

                    infect_times_history.append(new_current_infec_times)

                    if new_rec == 1:
                        # The infection becomes recovered
                        new_current_recovered = current_recovered + \
                            [current_infections[elim_infec]]
                        recov_history.append(new_current_recovered)

                        new_current_recov_infec_times = \
                            current_recov_infec_times + \
                            [current_infec_times[elim_infec]]
                        recov_infect_times_history.append(
                            new_current_recov_infec_times)
                    else:
                        # The infection dies
                        recov_history.append(current_recovered)
                        recov_infect_times_history.append(
                            current_recov_infec_times)

                # If a new infection occurs in the step
                elif new_infec == 1:
                    # Read in the last structure of infections and add new
                    # infection to the timeline
                    current_susceptibles = susc_history[-1][:-1]
                    current_infections = infect_history[-1] + \
                        [susc_history[-1][-1]]
                    current_recovered = recov_history[-1]
                    current_infec_times = infect_times_history[-1] + \
                        [float(current_time)]
                    current_recov_infec_times = recov_infect_times_history[-1]

                    susc_history.append(current_susceptibles)
                    infect_history.append(current_infections)
                    recov_history.append(current_recovered)

                    infect_times_history.append(current_infec_times)
                    recov_infect_times_history.append(
                            current_recov_infec_times)

                    infec_incidence.append(infec_incidence[-1]+[1])
                # If no change in infections occurs in the step
                else:
                    infec_incidence.append(infec_incidence[-1])
                    # Read in the last structure of infections
                    current_susceptibles = susc_history[-1]
                    current_infections = infect_history[-1]
                    current_recovered = recov_history[-1]
                    current_infec_times = infect_times_history[-1]
                    current_recov_infec_times = recov_infect_times_history[-1]

                    if new_susc == 1:
                        # New suceptible
                        current_susceptibles.append(self.last_used_id + 1)
                        self.last_used_id += 1
                    if new_susc == -1:
                        # A susceptible dies
                        current_susceptibles = current_susceptibles[:-1]

                    if new_rec == -1:
                        # A recovered dies
                        current_recovered = current_recovered[:-1]
                        current_recov_infec_times = \
                            current_recov_infec_times[:-1]

                    susc_history.append(current_susceptibles)
                    infect_history.append(current_infections)
                    recov_history.append(current_recovered)

                    infect_times_history.append(current_infec_times)
                    recov_infect_times_history.append(
                            current_recov_infec_times)
            else:
                susc_history.append([1+id for id in range(i_S)])
                infect_history.append([1+id for id in range(i_S, i_S+i_I)])
                recov_history.append(
                    [1+id for id in range(i_S+i_I, i_S+i_I+i_R)])

                infec_incidence.append([])

                infect_times_history.append(
                    [0 for _ in range(i_S, i_S+i_I)])
                recov_infect_times_history.append(
                    [0 for _ in range(i_S+i_I, i_S+i_I+i_R)])

            tau, i_S, i_I, i_R, new_susc, new_infec, new_rec, i_P = \
                self.one_step_gillespie(
                    current_time + self._cal_delay, i_S, i_I, i_R, i_P)

            current_time += tau

        self.infect_incidence = np.zeros(interval)
        # Keep only integer timepoints solutions
        for t in range(interval):
            pos = np.where(np.asarray(time_solution) <= times[t])
            solution[t, :] = large_solution[pos[-1][-1]]

            S_history.append(susc_history[pos[-1][-1]])
            I_history.append(infect_history[pos[-1][-1]])
            R_history.append(recov_history[pos[-1][-1]])

            I_times_history.append(infect_times_history[pos[-1][-1]])
            R_times_history.append(recov_infect_times_history[pos[-1][-1]])

            if t > 0:
                previous_pos = np.where(
                    np.asarray(time_solution) <= times[t-1])

                self.infect_incidence[t] = len(infec_incidence[
                    pos[-1][-1]]) - len(infec_incidence[previous_pos[-1][-1]])

        return (solution, S_history, I_history, R_history,
                I_times_history, R_times_history)

    def simulate_fixed_times(
            self, parameters, start_time, end_time, calendar_date=None):
        r"""
        Computes the number of each type of individuals in the population
        between the given time points.

        Parameters
        ----------
        parameters
            (list) List of quantities that characterise the epidemic dynamics
            in this order: the initial counts for each compartment (i_S, i_I,
            i_R, i_P), the birth rate of susceptibles, the death rate on
            suscepible (and recovered, :math:`\mu`) and the infectious
            individuals (:math:`\nu`), the transmission rate (:math:`\beta`)
            and the recovery rate (:math:`\gamma`) respectively.
        start_time
            (int) Time from which we start the simulation of the population.
        end_time
            (int) Time at which we end the simulation of the population.
        calendar_date
            (int) Calendar date from beginning of the year when simulation is
            started

        """
        # Check correct format of output
        self._check_times(start_time, end_time)

        self._check_parameters_format(parameters)
        self._set_parameters(parameters)

        # Determine calendar date delay in birth rate timeline
        if calendar_date is None:
            self._cal_delay = 0
        else:
            self._cal_delay = calendar_date

        (sol, S_history, I_history, R_history,
         I_times_history, R_times_history) = \
            self.gillespie_algorithm_fixed_times(start_time, end_time)

        output = sol

        return (output, S_history, I_history, R_history,
                I_times_history, R_times_history)

    def _set_parameters(self, parameters):
        """
        Split parameters into the features of the model.

        """
        # initial conditions
        self.init_cond = parameters[:4]
        self.N = sum(self.init_cond[:3])  # total prey population

        # birth rates
        if isinstance(parameters[4], (float, int)):
            # Same birth rate for every day of the year
            self.theta = mvr.constant_func(parameters[4])
        else:
            self.theta = parameters[4]

        # death rates
        mu, nu = parameters[5:7]
        if isinstance(mu, (float, int)):
            # Same birth rate for every day of the year
            self.mu_S = mvr.constant_func(parameters[5])
        else:
            self.mu_S = mu

        if isinstance(nu, (float, int)):
            # Same birth rate for every day of the year
            self.mu_I = mvr.constant_func(parameters[6])
        else:
            self.mu_I = nu

        # transition rates
        self.beta = parameters[7]
        self.gamma = parameters[8]

        # predator rates
        pred_birth, pred_death_H, pred_death_L = parameters[9:12]
        if isinstance(pred_birth, (float, int)):
            # Same birth rate for every day of the year
            self.pred_birth = mvr.constant_func(pred_birth)
        else:
            self.pred_birth = pred_birth

        if isinstance(pred_death_H, (float, int)):
            # Same death rate for every day of the year
            self.pred_death_H = mvr.constant_func(pred_death_H)
        else:
            self.pred_death_H = pred_death_H

        if isinstance(pred_death_L, (float, int)):
            # Same death rate for every day of the year
            self.pred_death_L = mvr.constant_func(pred_death_L)
        else:
            self.pred_death_L = pred_death_L

        # predation rates
        predation_S, predation_I = parameters[12:]
        if isinstance(predation_S, (float, int)):
            # Same predation rate for every day of the year
            self.predation_S = mvr.constant_func(predation_S)
        else:
            self.predation_S = predation_S

        if isinstance(predation_I, (float, int)):
            # Same predation rate for every day of the year
            self.predation_I = mvr.constant_func(predation_I)
        else:
            self.predation_I = predation_I

    def _check_parameters_format(self, parameters):
        """
        Checks the format of the `paramaters` input in the simulation methods.

        """
        if not isinstance(parameters, list):
            raise TypeError('Parameters must be given in a list format.')
        if len(parameters) != 14:
            raise ValueError('List of parameters needs to be of length 14.')
        for _ in range(4):
            if not isinstance(parameters[_], int):
                raise TypeError(
                    'Initial compartment count must be integer.')
            if parameters[_] < 0:
                raise ValueError('Initial compartment count must be => 0.')

        # Check the birth rate format
        if not isinstance(parameters[4], (float, int)) and not hasattr(
                parameters[4], '__call__'):
            raise TypeError(
                'Birth rate must be integer, float, or a function.')
        if isinstance(parameters[4], (float, int)) and parameters[4] < 0:
            raise ValueError('Birth rate must be => 0.')

        # Check the death rate format
        for _ in range(5, 7):
            if not isinstance(parameters[_], (float, int)) and not hasattr(
                    parameters[_], '__call__'):
                raise TypeError(
                    'Death rate must be integer float, or a function.')
            if isinstance(parameters[_], (float, int)) and parameters[_] < 0:
                raise ValueError('Death rate must be => 0.')

        for _ in range(7, 9):
            if not isinstance(parameters[_], (float, int)):
                raise TypeError(
                    'Transition rate must be integer or float.')
            if parameters[_] < 0:
                raise ValueError('Transition rate must be => 0.')

        # Check the predator rates format
        for _ in range(9, 12):
            if not isinstance(parameters[_], (float, int)) and not hasattr(
                    parameters[_], '__call__'):
                raise TypeError(
                    'Predator rate must be integer float, or a function.')
            if isinstance(parameters[_], (float, int)) and parameters[_] < 0:
                raise ValueError('Predator rate must be => 0.')

        # Check the predation rates format
        for _ in range(12, 14):
            if not isinstance(parameters[_], (float, int)) and not hasattr(
                    parameters[_], '__call__'):
                raise TypeError(
                    'Predation rate must be integer float, or a function.')
            if isinstance(parameters[_], (float, int)) and parameters[_] < 0:
                raise ValueError('Predation rate must be => 0.')

    def _compute_pred_birth(self, t_cal, i_P):
        """
        Returns the corresponding value of the birth rate of the
        predators according to the calendar date.

        Parameters
        ----------
        t_cal
            (int) current time according to the calendar date.
        i_P
            (int) current number of predators.

        """
        if (self.N >= self.N_crit) and self._is_summer(t_cal):
            # Find current integer day according to the calendar
            birth_rate = self.pred_birth(np.floor(t_cal).astype(int))
            return birth_rate * i_P * (1 - self.q(t_cal) * i_P / self.N)
        else:
            return 0

    def _compute_pred_death(self, t_cal, i_P):
        """
        Returns the corresponding value of the death rate of the
        predators according to the calendar date.

        Parameters
        ----------
        t_cal
            (int) current time according to the calendar date.
        i_P
            (int) current number of predators.

        """
        if (self.N < self.N_crit):
            # Find current integer day according to the calendar
            death_rate = self.pred_death_H(np.floor(t_cal).astype(int))
            return death_rate * i_P
        elif not self._is_summer(t_cal):
            # Find current integer day according to the calendar
            death_rate = self.pred_death_L(np.floor(t_cal).astype(int))
            return death_rate * i_P
        else:
            return 0

    def _compute_predation_S(self, t_cal, i_P):
        """
        Returns the corresponding value of the death rate of the
        non-infected prey due to predators according to the calendar date.

        Parameters
        ----------
        t_cal
            (int) current time according to the calendar date.
        i_P
            (int) current number of predators.

        """
        # Find current integer day according to the calendar
        predation_rate = self.predation_S(np.floor(t_cal).astype(int))

        return self._compute_mu_S(t_cal) + predation_rate * i_P / (
            self.N + self.d(t_cal))

    def _compute_predation_I(self, t_cal, i_P):
        """
        Returns the corresponding value of the death rate of the
        infected prey due to predators according to the calendar date.

        Parameters
        ----------
        t_cal
            (int) current time according to the calendar date.
        i_P
            (int) current number of predators.

        """
        # Find current integer day according to the calendar
        predation_rate = self.predation_I(np.floor(t_cal).astype(int))

        return self._compute_mu_I(t_cal) + predation_rate * i_P / (
            self.N + self.d(t_cal))

    def _is_summer(self, t_cal):
        """
        Returns a boolean of whether the calender date falls within the
        'summer' season or not.

        Parameters
        ----------
        t_cal
            (int) current time according to the calendar date.

        """
        if np.sin(2*math.pi*(np.floor(t_cal/7)/52 - self.rho)):
            return False
        else:
            return True
