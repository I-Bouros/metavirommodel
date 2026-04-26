# This file is part of metavirommodel
# (https://github.com/I-Bouros/metavirommodel)
# which is released under the BSD 3-clause license. See accompanying LICENSE.md
# for copyright notice and full license details.
#
"""
This script contains code for parameter inference of the rodent disease
dynamics model with a constant birth term for rodents (subject to temporal
variations in the growth rate) when Viral Read data is used for the
log-likelihood computation.

It uses a stochastic version of the standard SIR model with births and deaths.

"""

import numpy as np
import pandas as pd
import pints
from scipy.stats import gumbel_r

from scipy.integrate import solve_ivp

import metavirommodel as mvr
import metavirommodel.inference as mvri


#
# MVRVirReadLogLik Class
#

class MVRVirReadLogLik(pints.LogLikelihood):
    """MVRVirReadLogLik Class:
    Controller class to construct the log-likelihood needed for optimisation or
    inference of the MVR model with constant growth in a PINTS framework.

    Parameters
    ----------
    model : Metaviromodel
        The model for which we solve the optimisation or inference problem.
    viral_read_data: pandas.DataFrame
        Dataframe of the viral read count data, organised by individual ID
        and time of sample collection. Ordered by time of sample collection.
    parameters_vl : list of numpy.array
        List of parameters governing the viral read count model
        dynamics.
    generation_times: numpy.array or list
        List of probabilities of observing a detercatble Ct value t days after
        infection.

    """
    def __init__(self, model, viral_read_data, parameters_vl,
                 generation_times):
        # Set the prerequisites for the inference wrapper
        # Model and ICs data
        self._model = model

        # Viral read data
        self._vr_sampled_id = viral_read_data['ID'].values.tolist()
        self._vr_sampled_times = \
            viral_read_data['TimeOfSample'].values.tolist()
        self._vr_sampled_values = viral_read_data['Value'].values.tolist()

        self._vr_model_parameters = parameters_vl

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
        return 2

    def _probability_vr(self, vr_value, theta, t):
        """
        Computes the probability of observing a specific viral read count at
        time t, using previous population infection frequencies, generation
        times and probabilities of viral read detectablity.

        Parameters
        ----------
        vr_value
            (float or int) Observed viral read count for which we compute its
            probability of observation.
        theta
            (1D numpy array) contains frequency of infection in the population
            in each time unit (usually days) including zeros.
        t
            evaluation time
        """
        # Compute vector of dectability of the ct_value a days after infection
        p_vector = np.asarray([self.__compute_vr_detectable_pcr(
            vr_value, a) for a in np.arange(1, self._A_max+1)])

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

    def __compute_vr_detectable_pcr(self, vr_value, a):
        r"""
        Sample the corresponding viral read for an infected individual with
        respect to its time since infection.

        Parameters
        ----------
        vr_value
            (int or float) Observed viral read coubt.
        a
            (float) time since infection of the individuals for which we
            observe its viral read count.

        """
        # Read times of main points of behaviour change
        t_eclipse, t_peak, t_switch, t_mod, t_LOD = \
            self._vr_model_parameters[:5]

        # Read viral read count values associated with main points of
        # behaviour change
        v_zero, v_peak, v_switch, v_LOD = self._vr_model_parameters[5:9]

        # Read scale-specific parameters
        s_mod, sigma_obs = self._vr_model_parameters[9:]

        # Identify current value of the first distribution parameter
        v_mode_t = self.__compute_mode_vr_model(
            a, t_eclipse, t_peak, t_switch, t_LOD,
            np.log(v_zero), np.log(v_peak), np.log(v_switch), np.log(v_LOD))

        # Identify current value of the second distribution parameter
        sigma_t = self.__compute_sigma_vr_model(
            a, t_eclipse, t_peak, t_switch, t_mod, s_mod, sigma_obs)

        # Compute the normalisning constant P(VR > V_LOD)
        normalising_constant = 1 - gumbel_r.cdf(v_LOD, v_mode_t, sigma_t)

        # Compute log-likelihood of viral read value from from a Gumbel dist
        # VR ~ (V_mode_t, sigma_t)
        return gumbel_r.pdf(
            np.log(vr_value), v_mode_t, sigma_t)/normalising_constant

    def __compute_mode_vr_model(self, t, t_eclipse, t_peak, t_switch, t_LOD,
                                v_zero, v_peak, v_switch, v_LOD):
        """
        Compute the mode of the probability distribution used to determine
        observed viral read counts based on the time since infection.

        """
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

        return v_mode_t

    def __compute_sigma_vr_model(self, t, t_eclipse, t_peak, t_switch, t_mod,
                                 s_mod, sigma_obs):
        """
        Compute the variance of the probability distribution used to determine
        observed viral read counts based on the time since infection.

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
        times = np.arange(1, max(self._vr_sampled_times)+1)
        output = self._run_sir_model(
            parameters, times)

        # Incidence of infection
        n_incidence = self._new_infections(output, parameters, times)

        # Determine fractions of incidence of infection
        theta = np.divide(n_incidence, np.sum(output, axis=1))

        total_log_lik = 0

        # Compute log-likelihood
        try:
            # Log-likelihood contribution from viral read count data
            # collected at time t
            for t, time in enumerate(self._vr_sampled_times):
                if self._vr_sampled_values[t] > self._vr_model_parameters[8]:
                    # If sampled Viral read > V_LOD
                    total_log_lik += np.log(self._probability_vr(
                        vr_value=self._vr_sampled_values[t],
                        theta=theta,
                        t=time))

                else:
                    # If sampled Viral read <= V_LOD
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
# MVRVirReadInfer Class
#

class MVRVirReadInfer(object):
    """MVRVirReadInfer Class:
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
        super(MVRVirReadInfer, self).__init__()

        # Assign model for inference or optimisation
        if not isinstance(model, mvr.Metaviromodel):
            raise TypeError('Wrong model type for parameters inference.')

        self._model = model
        self._generation_times = generation_times

    def read_viral_read_data(
            self, viral_read_data, parameters_vl):
        """
        Sets the serology data used for the model's parameters inference.

        Parameters
        ----------
        viral_read_data: pandas.DataFrame
            Dataframe of the viral read count data, organised by individual ID
            and time of sample collection. Ordered by time of sample
            collection.
        parameters_vl : list of numpy.array
            List of parameters governing the viral read count model
            dynamics.

        """
        if not issubclass(type(viral_read_data), pd.DataFrame):
            raise TypeError(
                'Viral read data must use a Dataframe storage format.')
        if ('ID' not in viral_read_data.columns) and (
                'TimeOfSample' not in viral_read_data.columns) and (
                    'Value' not in viral_read_data.columns):
            raise TypeError(
                'Viral read data labels do not match prescribed names.')

        self._viral_read_counts = viral_read_data
        self._vl_parameters = parameters_vl

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
        loglikelihood = MVRVirReadLogLik(
            self._model, self._viral_read_counts, self._vl_parameters,
            self._generation_times)
        return loglikelihood(x)

    def _create_posterior(self):
        """
        Runs the initial conditions optimisation routine for the MVR model.

        """
        # Create a likelihood
        self.loglikelihood = MVRVirReadLogLik(
            self._model, self._viral_read_counts, self._vl_parameters,
            self._generation_times)

        # Create a prior
        log_prior = mvri.MVRCtValLogPrior(self._model)

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
