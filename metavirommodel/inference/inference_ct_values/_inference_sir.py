#
# This file is part of metavirommodel
# (https://github.com/I-Bouros/metavirommodel)
# which is released under the BSD 3-clause license. See accompanying LICENSE.md
# for copyright notice and full license details.
#
"""
This script contains code for parameter inference of the rodent disease
dynamics model with a constant birth term for rodents (subject to temporal
variations in the growth rate) when Ct Value data is used for the
log-likelihood computation.

It uses a stochastic version of the standard SIR model with births and deaths.

"""

import numpy as np
import pandas as pd
import pints
from scipy.stats import gumbel_r, uniform

from scipy.integrate import solve_ivp

import metavirommodel as mvr


#
# MVRCtValLogLik Class
#

class MVRCtValLogLik(pints.LogLikelihood):
    """MVRCtValLogLik Class:
    Controller class to construct the log-likelihood needed for optimisation or
    inference of the MVR model with constant growth in a PINTS framework.

    Parameters
    ----------
    model : Metaviromodel
        The model for which we solve the optimisation or inference problem.
    ct_values_data: pandas.DataFrame
        Dataframe of the Ct value data, organised by individual ID
        and time of sample collection. Ordered by time of sample collection.
    parameters_ct : list of numpy.array
        List of parameters governing the Ct value model dynamics.
    generation_times: numpy.array or list
        List of probabilities of observing a detercatble Ct value t days after
        infection.

    """
    def __init__(self, model, ct_values_data, parameters_ct, generation_times):
        # Set the prerequisites for the inference wrapper
        # Model and ICs data
        self._model = model

        # Serology data
        self._ct_sampled_id = ct_values_data['ID'].values.tolist()
        self._ct_sampled_times = ct_values_data['TimeOfSample'].values.tolist()
        self._ct_sampled_values = ct_values_data['Value'].values.tolist()

        self._ct_model_parameters = parameters_ct

        # Invert order of generation times for ease of computation
        self._gen_times = np.asarray(generation_times)[::-1]

        self._A_max = np.shape(self._gen_times)[0]

    def n_parameters(self):
        """
        Returns number of parameters for log-likelihood object.

        Returns
        -------
        int
            Number of parameters for log-likelihood object.

        """
        # return max(self._ct_sampled_times)
        return 2

    def _probability_ct(self, ct_value, theta, t):
        """
        Computes the probability of observing a specific ct value at time t,
        using previous population infection frequencies, generation times and
        probabilities of PCR detectablity.

        Parameters
        ----------
        ct_value
            (float or int) Observed value of Ct for which we compute its
            probability of observation.
        theta
            (1D numpy array) contains frequency of infection in the population
            in each time unit (usually days) including zeros.
        t
            evaluation time
        """
        # Compute vector of dectability of the ct_value a days after infection
        p_vector = np.asarray([self.__compute_ct_detectable_pcr(
            ct_value, a) for a in np.arange(1, self._A_max+1)])

        # Invert order for ease of computation
        p_vector = p_vector[::-1]

        if t > len(self._gen_times):
            start_date = t - len(self._gen_times) - 1
            prob = (
                theta[start_date:(t-1)] * self._gen_times *
                p_vector).sum()
            return prob

        prob = (
            theta[:(t-1)] * self._gen_times[-(t-1):] *
            p_vector[-(t-1):]).sum()

        return prob

    def _probability_detectable(self, theta, t):
        """
        Computes the probability of a randomly selected individuals being PCR
        detectable at time t, using previous population infection frequencies
        and generation times.

        Parameters
        ----------
        ct_value
            (float or int) Observed value of Ct for which we compute its
            probability of observation.
        theta
            (1D numpy array) contains frequency of infection in the population
            in each time unit (usually days) including zeros.
        t
            evaluation time
        """

        if t > len(self._gen_times):
            start_date = t - len(self._gen_times) - 1
            prob = (
                theta[start_date:(t-1)] * self._gen_times).sum()
            return prob

        prob = (
            theta[:(t-1)] * self._gen_times[-(t-1):]).sum()

        return prob

    def __compute_ct_detectable_pcr(self, Ct_value, a):
        r"""
        Sample the corresponding Ct value for an infected individual with
        respect to its time since infection.

        Parameters
        ----------
        Ct_value
            (int or float) Observed Ct value.
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
        a
            (float) time since infection of the individuals for which we
            observe its Ct value.

        """
        # Read times of main points of behaviour change
        t_eclipse, t_peak, t_switch, t_mod, t_LOD = \
            self._ct_model_parameters[:5]

        # Read Ct values associated with main points of
        # behaviour change
        c_zero, c_peak, c_switch, c_LOD = self._ct_model_parameters[5:9]

        # Read scale-specific parameters
        s_mod, sigma_obs = self._ct_model_parameters[9:]

        # Identify current value of the first distribution parameter
        c_mode_t = self.__compute_mode_ct_model(
            a, t_eclipse, t_peak, t_switch, t_LOD,
            c_zero, c_peak, c_switch, c_LOD)

        # Identify current value of the second distribution parameter
        sigma_t = self.__compute_sigma_ct_model(
            a, t_eclipse, t_peak, t_switch, t_mod, s_mod, sigma_obs)

        # Compute the normalisning constant P(0<Ct<C_LOD)
        normalising_constant = gumbel_r.cdf(c_LOD, c_mode_t, sigma_t) - \
            gumbel_r.cdf(0, c_mode_t, sigma_t)

        # Compute log-likeliooh of viral read value from from a Gumbel dist
        # Ct ~ (C_mode_t, sigma_t)
        return gumbel_r.pdf(Ct_value, c_mode_t, sigma_t)/normalising_constant

    def __compute_mode_ct_model(self, t, t_eclipse, t_peak, t_switch, t_LOD,
                                c_zero, c_peak, c_switch, c_LOD):
        """
        Compute the mode of the probability distribution used to determine
        observed Ct value based on the time since infection.

        """
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

        return c_mode_t

    def __compute_sigma_ct_model(self, t, t_eclipse, t_peak, t_switch, t_mod,
                                 s_mod, sigma_obs):
        """
        Compute the variance of the probability distribution used to determine
        observed Ct value based on the time since infection.

        """
        if (t < (t_eclipse + t_peak + t_switch)):
            sigma_t = sigma_obs
        elif (((t_eclipse + t_peak + t_switch) <= t) and (t < (
                t_eclipse + t_peak + t_switch + t_mod))):
            sigma_t = sigma_obs * (1 - ((1 - s_mod) / t_mod) * (
                t - t_eclipse - t_peak - t_switch))
        elif (((t_eclipse + t_peak + t_mod) <= t)):
            sigma_t = sigma_obs * s_mod

        return sigma_t

    def __right_hand_side(self, t, y, c):
        r"""
        Constructs the RHS of the equations of the system of ODEs for given a
        region and time point.

        Parameters
        ----------
        t : float
            Time point at which we compute the evaluation.
        y : numpy.array
            Array of all the compartments of the ODE system. It assumes
            y = [S, I, R] where each letter actually refers to all compartment
            of that type. (e.g. S refers to the compartments of susceptibles).
        c : list
            List of values used to compute the parameters of the ODEs
            system. It assumes c = [beta, gamma].

        Returns
        -------
        numpy.array
            Age-structured matrix representation of the RHS of the ODEs system.

        """
        # Split compartments into their types
        s, i, _ = y

        # Read parameters of the system
        theta, mu, nu, beta, gamma = c

        # Write actual RHS
        dydt = [
            theta(t) - beta * np.asarray(s * i) / np.sum(y) - mu * np.asarray(
                s),
            beta * np.asarray(s * i) / np.sum(y) - gamma * np.asarray(
                i) - nu(t) * np.asarray(i),
            gamma * np.asarray(i) - mu * np.asarray(_)]

        return dydt

    def __scipy_solver(self, times):
        """
        Computes the values in each compartment of the ODEs system using
        the 'off-the-shelf' solver of the IVP from :module:`scipy`.

        Parameters
        ----------
        times : list
            List of time points at which we wish to evaluate the ODEs system.

        Returns
        -------
        dict
            Solution of the ODE system at the time points provided.

        """
        # Initial conditions
        init_cond = np.asarray(self._y_init).tolist()

        # Solve the system of ODEs
        sol = solve_ivp(
            lambda t, y: self.__right_hand_side(t, y, self._c),
            [times[0], times[-1]], init_cond, t_eval=times)
        return sol

    def _new_infections(self, output, c, times):
        """
        Computes number of new infections at each time step in specified
        region, given the simulated timeline of susceptible number of
        individuals.

        Parameters
        ----------
        output : numpy.array
            Age-structured output of the simulation method for the
            determinsitic SIR model.

        Returns
        -------
        numpy.array
            Age-structured matrix of the number of new infections from the
            simulation method for the determinsitic SIR model.

        Notes
        -----
        Always run :meth:`_run_sir_model` before running this one.

        """
        beta = c[6]
        d_infec = np.empty(times.shape[0])

        for ind, t in enumerate(times.tolist()):
            # Read from output
            s = output[ind, 0]
            i = output[ind, 1]

            # fraction of new infectives in delta_t time step
            d_infec[ind] = beta * np.asarray(s * i) / np.sum(output[ind, :])

            if np.any(d_infec[ind] < 0):  # pragma: no cover
                d_infec[ind] = np.zeros_like(d_infec[ind])

        return d_infec

    def _run_sir_model(self, parameters, times):
        """
        """
        # Split parameters into the features of the model
        self._y_init = parameters[:3]
        self._c = parameters[3:]

        self._times = np.asarray(times)

        # Select method of simulation
        sol = self.__scipy_solver(times)

        output = sol['y']

        return output.transpose()

    def _log_likelihood(self, var_parameters):
        """
        Computes the log-likelihood of the non-fixed parameters
        using death and serology data.

        Parameters
        ----------
        var_parameters : list
            List of varying parameters of the model for which
            the log-likelihood is computed for.

        Returns
        -------
        float
            Value of the log-likelihood for the given choice of
            free parameters.

        """
        # Run SIR model
        parameters = self._model.init_cond + [
            self._model.theta, var_parameters[1], self._model.mu_I,
            var_parameters[0] * self._model.gamma, self._model.gamma]
        times = np.arange(1, max(self._ct_sampled_times)+1)
        output = self._run_sir_model(
            parameters, times)

        # Incidence of infection
        n_incidence = self._new_infections(output, parameters, times)

        # Determine fractions of incidence of infection
        theta = np.divide(n_incidence, np.sum(output, axis=1))

        total_log_lik = 0

        # Compute log-likelihood
        try:
            # Log-likelihood contribution from Ct values data
            # collected at time t
            for t, time in enumerate(self._ct_sampled_times):
                if self._ct_sampled_values[t] < self._ct_model_parameters[8]:
                    # If sampled Ct value < C_LOD
                    total_log_lik += np.log(self._probability_ct(
                        ct_value=self._ct_sampled_values[t],
                        theta=theta,
                        t=time))

                else:
                    # If sampled Ct value >= C_LOD
                    total_log_lik += np.log(1 - self._probability_detectable(
                        theta=theta,
                        t=time))

            return np.sum(total_log_lik)

        except ValueError:  # pragma: no cover
            return -np.inf

    def __call__(self, x):
        """
        Evaluates the log-likelihood in a PINTS framework.

        Parameters
        ----------
        x : list
            List of free parameters used for computing the log-likelihood.

        Returns
        -------
        float
            Value of the log-likelihood at the given point in the free
            parameter space.

        """
        return self._log_likelihood(x)


#
# MVRCtValLogPrior Class
#

class MVRCtValLogPrior(pints.LogPrior):
    """MVRCtValLogPrior Class:
    Controller class to construct the log-prior needed for optimisation or
    inference of the MVR model with constant growth in a PINTS framework.

    Parameters
    ----------
    model : Metaviromodel
        The model for which we solve the optimisation or inference problem.

    """
    def __init__(self, model):
        super(MVRCtValLogPrior, self).__init__()
        # Set the prerequisites for the inference wrapper
        # Model
        self._model = model

    def n_parameters(self):
        """
        Returns number of parameters for log-prior object.

        Returns
        -------
        int
            Number of parameters for log-prior object.

        """
        # return self._times
        return 2

    def __call__(self, x):
        """
        Evaluates the log-prior in a PINTS framework.

        Parameters
        ----------
        x : list
            List of free parameters used for computing the log-prior.

        Returns
        -------
        float
            Value of the log-prior at the given point in the free
            parameter space.

        """
        # Prior contribution of R0 and death rate
        return uniform.logpdf(x[0], 1.05, 5) + uniform.logpdf(
            x[1], 0.001, 0.01)


#
# MVRCtValInfer Class
#

class MVRCtValInfer(object):
    """MVRCtValInfer Class:
    Controller class for the optimisation or inference of parameters of the
    MVR model with constant growth in a PINTS framework.

    Parameters
    ----------
    model : Metaviromodel
        The model for which we solve the optimisation or inference problem.
    generation_times: numpy.array or list
        List of probabilities of observing a detercatble Ct value t days after
        infection.

    """
    def __init__(self, model, generation_times):
        super(MVRCtValInfer, self).__init__()

        # Assign model for inference or optimisation
        if not isinstance(model, mvr.Metaviromodel):
            raise TypeError('Wrong model type for parameters inference.')

        self._model = model
        self._generation_times = generation_times

    def read_ct_values_data(
            self, ct_values_data, parameters_ct):
        """
        Sets the serology data used for the model's parameters inference.

        Parameters
        ----------
        ct_values_data: pandas.DataFrame
            Dataframe of the Ct value data, organised by individual ID
            and time of sample collection. Ordered by time of sample
            collection.
        parameters_ct : list of numpy.array
            List of parameters governing the Ct value model dynamics.

        """
        if not issubclass(type(ct_values_data), pd.DataFrame):
            raise TypeError(
                'Ct values data must use a Dataframe storage format.')
        if ('ID' not in ct_values_data.columns) and (
                'TimeOfSample' not in ct_values_data.columns) and (
                    'Value' not in ct_values_data.columns):
            raise TypeError(
                'Ct values data labels do not match prescribed names.')

        self._ct_values = ct_values_data
        self._ct_parameters = parameters_ct

    def return_loglikelihood(self, x):
        """
        Return the log-likelihood used for the optimisation or inference.

        Parameters
        ----------
        x : list
            List of free parameters used for computing the log-likelihood.

        Returns
        -------
        float
            Value of the log-likelihood at the given point in the free
            parameter space.

        """
        loglikelihood = MVRCtValLogLik(
            self._model, self._ct_values, self._ct_parameters,
            self._generation_times)
        return loglikelihood(x)

    def _create_posterior(self):
        """
        Runs the initial conditions optimisation routine for the MVR model.

        """
        # Create a likelihood
        self.loglikelihood = MVRCtValLogLik(
            self._model, self._ct_values, self._ct_parameters,
            self._generation_times)

        # Create a prior
        log_prior = MVRCtValLogPrior(self._model)

        # Create a posterior log-likelihood (log(likelihood * prior))
        self._log_posterior = pints.LogPosterior(self.loglikelihood, log_prior)

    def inference_problem_setup(self, num_iter):
        """
        Runs the parameter inference routine for the MVR model.

        Parameters
        ----------
        num_iter : integer
            Number of iterations the MCMC sampler algorithm is run for.

        Returns
        -------
        numpy.array
            3D-matrix of the proposed parameters for each iteration for
            each of the chains of the MCMC sampler.

        """
        self._create_posterior()

        # Starting points using optimisation object
        x0 = [[3, 0.002], [2, 0.004], [3, 0.004]]

        # Create MCMC routine
        mcmc = pints.MCMCController(
            self._log_posterior, 3, x0)
        mcmc.set_max_iterations(num_iter)
        mcmc.set_log_to_screen(True)
        mcmc.set_parallel(True)

        print('Running...')
        chains = mcmc.run()
        print('Done!')

        param_names = ['R0', 'mu']

        # Check convergence and other properties of chains
        results = pints.MCMCSummary(
            chains=chains, time=mcmc.time(),
            parameter_names=param_names)
        print(results)

        return chains

    def optimisation_problem_setup(self):
        """
        Runs the initial conditions optimisation routine for the MVR model.

        Returns
        -------
        numpy.array
            Matrix of the optimised parameters at the end of the optimisation
            procedure.
        float
            Value of the log-posterior at the optimised point in the free
            parameter space.

        """
        self._create_posterior()

        # Starting points - random position in parameter hyperspace
        x0 = [3, 0.004]

        # Create optimisation routine
        optimiser = pints.OptimisationController(
            self._log_posterior, x0, method=pints.CMAES)

        optimiser.set_max_unchanged_iterations(100, 1)

        found_ics, found_posterior_val = optimiser.run()
        print(found_ics, found_posterior_val)

        print("Optimisation phase is finished.")

        return found_ics, found_posterior_val
