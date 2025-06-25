#
# This file is part of metavirommodel
# (https://github.com/I-Bouros/metavirommodel)
# which is released under the BSD 3-clause license. See accompanying LICENSE.md
# for copyright notice and full license details.
#

import numpy as np
import pints
from scipy.stats import uniform, gumbel_r, gumbel_l


class Metaviromodel(pints.ForwardModel):
    r"""Metaviromodel Class:
    Base class for the forward simulation of the epidemic transmission dynamic
    of a population of rodents.

    Three types of individuals are considered based on their seroligcal status
    - susceptible individuals (S), infectious (I) and recovered (R).

    Susceptible individuals are born at a constant rate and die at the same
    rate as those recovered. A different death rate due to diesease is
    considered for the infected individuals. A susceptible individuals goes on
    to become infected at a constant rate.

    The system of equations that describe the isolated possible events that can
    occur

    .. math::
        :nowrap:

        \begin{eqnarray}
            S  &\xrightarrow{\beta} I \\
            I  &\xrightarrow{\gamma} R \\
            \emptyset  &\xrightarrow{\theta} S \\
            S &\xrightarrow{\mu} \emptyset \\
            I &\xrightarrow{\nu} \emptyset \\
            R &\xrightarrow{\mu} \emptyset
        \end{eqnarray}

    where :math:`\mu` and :math:`\nu` are the rates of natural death in
    the susceptibles and recovered, and infectious respectively,
    :math:`\theta` is the birth rate in the susceptibles, :math:`\beta` is
    the transmission rate and :math:`\gamma` is the recovery rate.

    """
    def __init__(self):
        super(Metaviromodel, self).__init__()

        # Assign default values
        self._output_names = ['S', 'I', 'R']
        self._parameter_names = [
            'S0', 'I0', 'R0', 'theta', 'mu', 'nu', 'beta', 'gamma']

        # The default number of outputs is 3,
        # i.e. S, I and R
        self._n_outputs = len(self._output_names)
        # The default number of outputs is 8,
        # i.e. 3 initial conditions and 5 parameters
        self._n_parameters = len(self._parameter_names)

        self._output_indices = np.arange(self._n_outputs)

    def n_outputs(self):
        """
        Returns the number of outputs.

        """
        return self._n_outputs

    def n_parameters(self):
        """
        Returns the number of parameters.

        """
        return self._n_parameters

    def output_names(self):
        """
        Returns the (selected) output names.

        """
        names = [self._output_names[x] for x in self._output_indices]
        return names

    def parameter_names(self):
        """
        Returns the parameter names.

        """
        return self._parameter_names

    def set_outputs(self, outputs):
        """
        Checks existence of outputs.

        """
        for output in outputs:
            if output not in self._output_names:
                raise ValueError(
                    'The output names specified must be in correct forms')

        output_indices = []
        for output_id, output in enumerate(self._output_names):
            if output in outputs:
                output_indices.append(output_id)

        # Remember outputs
        self._output_indices = output_indices
        self._n_outputs = len(outputs)

    def _compute_theta(self, t_cal):
        """
        Returns the corresponding value of the birth rate of the
        susceptible according to the calendar date.

        Parameters
        ----------
        t_cal
            (int) current time according to the calendar date.

        """
        # Find current integer day according to the calendar
        return self.theta(np.floor(t_cal).astype(int))

    def _compute_mu_S(self, t_cal):
        """
        Returns the corresponding value of the death rate of the
        susceptible and recovered according to the calendar date.

        Parameters
        ----------
        t_cal
            (int) current time according to the calendar date.

        """
        # Find current integer day according to the calendar
        return self.mu_S(np.floor(t_cal).astype(int))

    def _compute_mu_I(self, t_cal):
        """
        Returns the corresponding value of the death rate of the
        infected individuals according to the calendar date.

        Parameters
        ----------
        t_cal
            (int) current time according to the calendar date.

        """
        # Find current integer day according to the calendar
        return self.mu_I(np.floor(t_cal).astype(int))

    def one_step_gillespie(self, t_cal, i_S, i_I, i_R):
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

        """
        # Generate random number for reaction and time to next reaction
        u, u1 = uniform.rvs(size=2)

        self.N = sum((i_S, i_I, i_R))

        new_infec = 0

        # Compute propensities
        if self.N > 0:
            propens_1 = self.beta * i_S * i_I / self.N
            propens_2 = self.gamma * i_I
            propens_3 = self._compute_theta(t_cal)
            propens_4 = self._compute_mu_S(t_cal) * i_S
            propens_5 = self._compute_mu_I(t_cal) * i_I
            propens_6 = self._compute_mu_S(t_cal) * i_R

            propens = np.array([
                propens_1, propens_2, propens_3, propens_4,
                propens_5, propens_6])
            sum_propens = np.empty(propens.shape)

            if np.sum(propens) > 0:
                for e in range(propens.shape[0]):
                    sum_propens[e] = np.sum(propens[:(e+1)]) / np.sum(propens)
                # Time to next reaction
                tau = np.log(1/u1) / np.sum(propens)

                if u < sum_propens[0]:
                    i_S += -1
                    i_I += 1
                    new_infec = 1
                elif (u >= sum_propens[0]) and (u < sum_propens[1]):
                    i_I += -1
                    i_R += 1
                    new_infec = -1
                elif (u >= sum_propens[1]) and (u < sum_propens[2]):
                    i_S += 1
                elif (u >= sum_propens[2]) and (u < sum_propens[3]):
                    i_S += -1
                elif (u >= sum_propens[3]) and (u < sum_propens[4]):
                    i_I += -1
                    new_infec = -1
                else:
                    i_R += -1

            else:
                tau = None

        return (tau, i_S, i_I, i_R, new_infec)

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
        times = np.arange(start_time, end_time+0.5, 1, dtype=np.integer)
        interval = end_time - start_time + 1

        # Split compartments into their types
        i_S, i_I, i_R = self.init_cond

        large_solution = []
        time_solution = []
        infect_history = [[1] * i_I]
        infect_times_history = [[0] * i_I]
        solution = np.empty((interval, 3), dtype=np.integer)
        I_history = []
        I_times_history = []
        current_time = start_time
        while current_time <= end_time:
            time_solution.append(float(current_time))
            large_solution.append([i_S, i_I, i_R])
            tau, i_S, i_I, i_R, new_infec = self.one_step_gillespie(
                current_time + self._cal_delay, i_S, i_I, i_R)

            # If there is a next reaction
            if tau is not None:
                # If an infection disappears
                if new_infec == -1:
                    # Read in the last structure of infections
                    current_infections = infect_history[-1]
                    current_infec_times = infect_times_history[-1]

                    # Select infection to disappear using a multinomial
                    # distribution
                    weights = current_time - np.asarray(current_infec_times)
                    if np.sum(weights) == 0:
                        elim_infec = np.random.choice(
                            range(sum(current_infections)))
                    else:
                        elim_infec = np.random.choice(
                            range(sum(current_infections)),
                            p=weights/np.sum(weights))

                    # Eliminate infection
                    current_infections.remove(current_infections[elim_infec])
                    current_infec_times.remove(current_infec_times[elim_infec])

                    infect_history.append(current_infections)
                    infect_times_history.append(current_infec_times)
                # If a new infection occurs in the step
                elif new_infec == 1:
                    # Read in the last structure of infections and add new
                    # infection to the timeline
                    current_infections = infect_history[-1] + [1]
                    current_infec_times = infect_times_history[-1] + \
                        [current_time]

                    infect_history.append(current_infections)
                    infect_times_history.append(current_infec_times)
                # If no change in infections occurs in the step
                else:
                    # Read in the last structure of infections
                    current_infections = infect_history[-1]
                    current_infec_times = infect_times_history[-1]

                    infect_history.append(current_infections)
                    infect_times_history.append(current_infec_times)

                current_time += tau

            else:
                # If there is no next reaction as we run out of individuals
                # Read in the last structure of infections
                current_infections = infect_history[-1]
                current_infec_times = infect_times_history[-1]

                infect_history.append(current_infections)
                infect_times_history.append(current_infec_times)

                current_time += 1

        # Keep only integer timepoints solutions
        for t in range(interval):
            pos = np.where(np.asarray(time_solution <= times[t]))
            solution[t, :] = large_solution[pos[-1][-1]]
            I_history.append(infect_history[1:][pos[-1][-1]])
            I_times_history.append(infect_times_history[1:][pos[-1][-1]])

        return solution, I_history, I_times_history

    def ct_model(self, parameters_ct, t):
        r"""
        Sample the corresponding Ct value for an infected individual with
        respect to its time since infection.

        Parameters
        ----------
        parameters_ct
            (list) List of parameters governing the Ct value model dynamics:
            the times from infection to initial viral growth (t_eclipse),
            from initial viral growth to peak viral load (t_peak), from peak
            viral load to secondary waning phase (t_switch), from secondary
            waning phase until Gumbel distribution reaches its minimum scale
            parameter (t_mod) and finally from infection until modal Ct value
            is equal to the limit of detection (t_LOD), the Ct values
            associated with the time of infection (c_zero), peak viral load
            (c_peak), the debut of the secondary waning phase at
            :math:`t_eclipse + t_peak + t_switch` (c_switch) and the limit
            of detection of Ct value (c_LOD), the multiplicative factor
            applied to scale parameter for the Gumbel distribution starting at
            time :math:`t_eclipse + t_peak + t_switch + t_scale` (s_mod), and
            the initial scale parameter for the Gumbel distribution until time
            :math:`t_eclipse + t_peak + t_switch` (sigma_obs) respectively.
        t
            (float) time since infection of the individuals for which we
            calculate its Ct value.

        """
        # Read times of main points of behaviour change
        t_eclipse, t_peak, t_switch, t_mod, t_LOD = parameters_ct[:5]

        # Read Ct values associated with main points of
        # behaviour change
        c_zero, c_peak, c_switch, c_LOD = parameters_ct[5:9]

        # Read scale-specific parameters
        s_mod, sigma_obs = parameters_ct[9:]

        # Identify current value of the first distribution parameter
        if t <= t_eclipse:
            c_mode_t = c_zero
        elif (t_eclipse < t) and (t <= t_eclipse + t_peak):
            c_mode_t = c_zero + (
                (c_peak - c_zero) / (t_peak)) * (t - t_eclipse)
        elif ((t_eclipse + t_peak) < t) and (
                t <= (t_eclipse + t_peak + t_switch)):
            c_mode_t = c_peak + ((c_switch - c_peak) / t_switch) * (
                t - t_eclipse - t_peak)
        elif ((t_eclipse + t_peak + t_switch) < t):
            c_mode_t = c_switch + ((c_LOD - c_switch) / (
                t_LOD - t_switch - t_peak - t_eclipse)) * (
                t - t_eclipse - t_peak - t_switch)

        # Identify current value of the second distribution parameter
        if (t < (t_eclipse + t_peak + t_switch)):
            sigma_t = sigma_obs
        elif (((t_eclipse + t_peak + t_switch) <= t) and (t < (
                t_eclipse + t_peak + t_switch + t_mod))):
            sigma_t = sigma_obs * (1 - ((1 - s_mod) / t_mod) * (
                t - t_eclipse - t_peak - t_switch))
        elif (((t_eclipse + t_peak + t_mod) <= t)):
            sigma_t = sigma_obs * s_mod

        # Draw Ct value from from a Gumbel dist Ct ~ (C_mode_t, sigma_t)
        Ct_value = gumbel_r.rvs(c_mode_t, sigma_t)

        if Ct_value > c_zero:
            Ct_value = c_zero  # capped if we get abnormal Ct value
        return Ct_value

    def viral_read_model(self, parameters_vl, t):
        r"""
        Sample the corresponding viral read count for an infected individual
        with respect to its time since infection.

        Parameters
        ----------
        parameters_vl
            (list) List of parameters governing the viral read count model
            dynamics: the times from infection to initial viral growth
            (t_eclipse), from initial viral growth to peak viral load (t_peak),
            from peak viral load to secondary waning phase (t_switch), from
            secondary waning phase until Gumbel distribution reaches its
            minimum scale parameter (t_mod) and finally from infection until
            modal viral read count is equal to the limit of detection (t_LOD),
            the viral read counts associated with the time of infection
            (v_zero), peak viral load (v_peak), the debut of the secondary
            waning phase at :math:`t_eclipse + t_peak + t_switch` (v_switch)
            and the limit of detection of viral read count (v_LOD), the
            multiplicative factor applied to scale parameter for the Gumbel
            distribution starting at time :math:`t_eclipse + t_peak + t_switch
            + t_scale` (s_mod), and the initial scale parameter for the Gumbel
            distribution until time :math:`t_eclipse + t_peak + t_switch`
            (sigma_obs) respectively.
        t
            (float) time since infection of the individuals for which we
            calculate its viral read count.

        """
        # Read times of main points of behaviour change
        t_eclipse, t_peak, t_switch, t_mod, t_LOD = parameters_vl[:5]

        # Read viral read count values associated with main points of
        # behaviour change
        v_zero, v_peak, v_switch, v_LOD = parameters_vl[5:9]

        # Read scale-specific parameters
        s_mod, sigma_obs = parameters_vl[9:]

        # Identify current value of the first distribution parameter
        if t <= t_eclipse:
            v_mode_t = v_zero
        elif (t_eclipse < t) and (t <= t_eclipse + t_peak):
            v_mode_t = v_zero + (
                (v_peak - v_zero) / (t_peak)) * (t - t_eclipse)
        elif ((t_eclipse + t_peak) < t) and (
                t <= (t_eclipse + t_peak + t_switch)):
            v_mode_t = v_peak + ((v_switch - v_peak) / t_switch) * (
                t - t_eclipse - t_peak)
        elif ((t_eclipse + t_peak + t_switch) < t):
            v_mode_t = v_switch + ((v_LOD - v_switch) / (
                t_LOD - t_switch - t_peak - t_eclipse)) * (
                t - t_eclipse - t_peak - t_switch)

        # Identify current value of the second distribution parameter
        if (t < (t_eclipse + t_peak + t_switch)):
            sigma_t = sigma_obs
        elif (((t_eclipse + t_peak + t_switch) <= t) and (t < (
                t_eclipse + t_peak + t_switch + t_mod))):
            sigma_t = sigma_obs * (1 - ((1 - s_mod) / t_mod) * (
                t - t_eclipse - t_peak - t_switch))
        elif (((t_eclipse + t_peak + t_mod) <= t)):
            sigma_t = sigma_obs * s_mod

        # Draw viral read value from from a Gumbel dist
        # VR ~ (V_mode_t, sigma_t)
        VR_value = gumbel_l.rvs(v_mode_t, sigma_t)

        if VR_value < 0:
            VR_value = 0  # capped if we get abnormal VR value
        return VR_value

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
            i_R), the birth rate of susceptibles, the death rate on suscepible
            (and recovered, :math:`\mu`) and the infectious individuals (
            :math:`\nu`), the transmission rate (:math:`\beta`) and the
            recovery rate (:math:`\gamma`) respectively.
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

        sol, I_history, I_times_history = self.gillespie_algorithm_fixed_times(
            start_time, end_time)

        output = sol

        return output, I_history, I_times_history

    def _check_times(self, start_time, end_time):
        """
        Checks format of start and end of simulation window.

        """
        if not isinstance(start_time, int):
            raise TypeError('Start time of siumlation must be integer.')
        if start_time <= 0:
            raise ValueError('Start time of siumlation must be > 0.')

        if not isinstance(end_time, int):
            raise TypeError('End time of siumlation must be integer.')
        if end_time <= 0:
            raise ValueError('Start time of siumlation must be > 0.')

        if start_time > end_time:
            raise ValueError('End time must be after start time.')

    def _set_parameters(self, parameters):
        """
        Split parameters into the features of the model.

        """
        # initial conditions
        self.init_cond = parameters[:3]
        self.N = sum(self.init_cond)

        # birth rates
        if isinstance(parameters[3], (float, int)):
            # Same birth rate for every day of the year
            self.theta = lambda _: parameters[3]
        else:
            self.theta = parameters[3]

        # death rates
        mu, nu = parameters[4:6]
        if isinstance(mu, (float, int)):
            # Same birth rate for every day of the year
            self.mu_S = lambda _: mu
        else:
            self.mu_S = mu

        if isinstance(nu, (float, int)):
            # Same birth rate for every day of the year
            self.mu_I = lambda _: nu
        else:
            self.mu_I = nu

        # transition rates
        self.beta = parameters[6]
        self.gamma = parameters[7]

    def _check_parameters_format(self, parameters):
        """
        Checks the format of the `paramaters` input in the simulation methods.

        """
        if not isinstance(parameters, list):
            raise TypeError('Parameters must be given in a list format.')
        if len(parameters) != 8:
            raise ValueError('List of parameters needs to be of length 8.')
        for _ in range(3):
            if not isinstance(parameters[_], int):
                raise TypeError(
                    'Initial compartment count must be integer.')
            if parameters[_] < 0:
                raise ValueError('Initial compartment count must be => 0.')

        # Check the birth rate format
        if not isinstance(parameters[3], (float, int)) and not hasattr(
                parameters[3], '__call__'):
            raise TypeError(
                'Birth rate must be integer, float, or a function.')
        if isinstance(parameters[3], (float, int)) and parameters[3] < 0:
            raise ValueError('Birth rate must be => 0.')

        # Check the death rate format
        for _ in range(4, 6):
            if not isinstance(parameters[_], (float, int)) and not hasattr(
                    parameters[_], '__call__'):
                raise TypeError(
                    'Death rate must be integer float, or a function.')
            if isinstance(parameters[_], (float, int)) and parameters[_] < 0:
                raise ValueError('Death rate must be => 0.')

        for _ in range(6, 8):
            if not isinstance(parameters[_], (float, int)):
                raise TypeError(
                    'Transition rate must be integer or float.')
            if parameters[_] < 0:
                raise ValueError('Transition rate must be => 0.')
