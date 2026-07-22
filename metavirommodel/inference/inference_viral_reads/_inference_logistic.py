#
# This file is part of metavirommodel
# (https://github.com/I-Bouros/metavirommodel)
# which is released under the BSD 3-clause license. See accompanying LICENSE.md
# for copyright notice and full license details.
#
"""
This script contains code for parameter inference of the rodent disease
dynamics model with a logistic birth term for rodents (subject to temporal
variations in the growth rate) when Viral Read data is used for the
log-likelihood computation.

It uses a stochastic version of the standard SIR model with births and deaths.

"""

import numpy as np
from scipy.integrate import solve_ivp
import pints

import metavirommodel as mvr
import metavirommodel.inference as mvri


#
# LogisticGrowthMVRVirReadLogLik Class
#

class LogisticGrowthMVRVirReadLogLik(mvri.MVRVirReadLogLik):
    """LogisticGrowthMVRVirReadLogLik Class:
    Controller class to construct the log-likelihood needed for optimisation or
    inference of the MVR model with logistic growth in a PINTS framework.

    Parameters
    ----------
    model : LogisticGrowthMetaviromodel
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
        super().__init__(
            model, viral_read_data, parameters_vl, generation_times)

        # Assign model for inference or optimisation
        if not isinstance(model, mvr.LogisticGrowthMetaviromodel):
            raise TypeError('Wrong model type for parameters inference.')

    def n_parameters(self):
        """
        Returns number of parameters for log-likelihood object.

        Returns
        -------
        int
            Number of parameters for log-likelihood object.

        """
        # return max(self._ct_sampled_times)
        return 1

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

        cc = self._model._carry_cap

        # Write actual RHS
        dydt = [
            theta(t) * np.sum(y) * (1 - np.sum(y) / cc(t)) - beta * np.asarray(
                s * i) / np.sum(y) - mu(t) * np.asarray(s),
            beta * np.asarray(s * i) / np.sum(y) - gamma * np.asarray(
                i) - nu(t) * np.asarray(i),
            gamma * np.asarray(i) - mu(t) * np.asarray(_)]

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
            self._model.theta, self._model.mu_S, self._model.mu_I,
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


#
# LogisticGrowthMVRVirReadInfer Class
#

class LogisticGrowthMVRVirReadInfer(mvri.MVRVirReadInfer):
    """LogisticGrowthMVRVirReadInfer Class:
    Controller class for the optimisation or inference of parameters of the
    MVR model with logistic growth in a PINTS framework.

    Parameters
    ----------
    model : LogisticGrowthMetaviromodel
        The model for which we solve the optimisation or inference problem.
    generation_times: numpy.array or list
        List of probabilities of observing a detercatble Ct value t days after
        infection.

    """
    def __init__(self, model, generation_times):
        super(LogisticGrowthMVRVirReadInfer, self).__init__(
            model, generation_times)

        # Assign model for inference or optimisation
        if not isinstance(model, mvr.LogisticGrowthMetaviromodel):
            raise TypeError('Wrong model type for parameters inference.')

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
        loglikelihood = LogisticGrowthMVRVirReadLogLik(
            self._model, self._viral_read_counts, self._vl_parameters,
            self._generation_times)
        return loglikelihood(x)

    def _create_posterior(self):
        """
        Runs the initial conditions optimisation routine for the MVR model.

        """
        # Create a likelihood
        self.loglikelihood = LogisticGrowthMVRVirReadLogLik(
            self._model, self._viral_read_counts, self._vl_parameters,
            self._generation_times)

        # Create a prior
        log_prior = mvri.LogisticMVRCtValLogPrior(self._model)

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
        x0 = [[3], [2], [2.5]]

        # Create MCMC routine
        mcmc = pints.MCMCController(
            self._log_posterior, 3, x0)
        mcmc.set_max_iterations(num_iter)
        mcmc.set_log_to_screen(True)
        mcmc.set_parallel(True)

        print('Running...')
        chains = mcmc.run()
        print('Done!')

        param_names = ['R0']

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
        x0 = [3]

        # Create optimisation routine
        optimiser = pints.OptimisationController(
            self._log_posterior, x0, method=pints.BareCMAES)

        optimiser.set_max_unchanged_iterations(100, 1)

        found_ics, found_posterior_val = optimiser.run()
        print(found_ics, found_posterior_val)

        print("Optimisation phase is finished.")

        return found_ics, found_posterior_val
