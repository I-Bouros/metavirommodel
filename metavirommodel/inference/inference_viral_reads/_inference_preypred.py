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

It uses a stochastic version of the standard SIR model with births and deaths
and prey-predator dynamics (Lotka-Volterra).

"""

import numpy as np
from scipy.integrate import solve_ivp
import pints

import metavirommodel as mvr
import metavirommodel.inference as mvri


#
# PreyPredGrowthMVRVirReadLogLik Class
#

class PreyPredGrowthMVRVirReadLogLik(mvri.LogisticGrowthMVRVirReadLogLik):
    """MVRVirReadInfer Class:
    Controller class for the optimisation or inference of parameters of the
    MVR model with logistic growth and prey-predator dynamics in a PINTS
    framework.

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
        super().__init__(
            model, viral_read_data, parameters_vl, generation_times)

        # Assign model for inference or optimisation
        if not isinstance(model, mvr.PreyPredMetaviromodel):
            raise TypeError('Wrong model type for parameters inference.')

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
        s, i, _, p = y

        # Read parameters of the system
        theta, mu, nu, beta, gamma, alpha, alpha_prime = c

        CC = self._model._carry_cap
        D = self._model

        # Write actual RHS
        dydt = [
            theta(t) * np.sum(y[:-1]) * (1 - np.sum(y[:-1]) /
                                         CC(t)) - beta * np.asarray(
                s * i) / np.sum(y[:-1]) - mu * np.asarray(s) - alpha(
                    t) * np.asarray(s * p) / (np.sum(y[:-1]) + D),
            beta * np.asarray(s * i) / np.sum(y[:-1]) - gamma * np.asarray(
                i) - nu * np.asarray(i) - alpha_prime(t) * np.asarray(
                    i * p) / (np.sum(y[:-1]) + D),
            gamma * np.asarray(i) - mu * np.asarray(_) - alpha(t) * np.asarray(
                    _ * p) / (np.sum(y[:-1]) + D),
            self._predator_rates(t, np.asarray(p))
            ]

        return dydt

    def _predator_rates(self, t, p):
        """
        Returns the ordinary differential equation asscociated with the
        ecological dynamics for the predator population.

        Parameters
        ----------
        t : float
            Time point at which we compute the evaluation.
        p : float
            Current number of predators.

        """

        return self._model._compute_pred_birth(t, p) - \
            self._model._compute_pred_death(t, p)

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
            determinsitic SIRP model.

        Returns
        -------
        numpy.array
            Age-structured matrix of the number of new infections from the
            simulation method for the determinsitic SIRP model.

        Notes
        -----
        Always run :meth:`_run_sir_model` before running this one.

        """
        beta = c[7]
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

    def _run_sirp_model(self, parameters, times):
        """
        """
        # Split parameters into the features of the model
        self._y_init = parameters[:4]
        self._c = parameters[4:]

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
        # Run SIRP model
        parameters = self._model.init_cond + [
            self._model.theta,
            self._model.mu_S, var_parameters[1] * self._model.mu_S,
            var_parameters[0] * self._model.gamma, self._model.gamma,
            self._model.predation_S,
            var_parameters[1] * self._model.predation_S]
        times = np.arange(1, max(self._vr_sampled_times)+1)
        output = self._run_sirp_model(
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
# PreyPredGrowthMVRVirReadInfer Class
#

class PreyPredGrowthMVRVirReadInfer(mvri.MVRVirReadInfer):
    """PreyPredGrowthMVRVirReadInfer Class:
    Controller class for the optimisation or inference of parameters of the
    MVR model with logistic growth and prey-predator dynamics in a PINTS
    framework.

    Parameters
    ----------
    model : LogisticGrowthMetaviromodel
        The model for which we solve the optimisation or inference problem.
    generation_times: numpy.array or list
        List of probabilities of observing a detercatble Ct value t days after
        infection.

    """
    def __init__(self, model, generation_times):
        super(PreyPredGrowthMVRVirReadInfer, self).__init__(
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
        loglikelihood = PreyPredGrowthMVRVirReadLogLik(
            self._model, self._viral_read_counts, self._vl_parameters,
            self._generation_times)
        return loglikelihood(x)

    def _create_posterior(self):
        """
        Runs the initial conditions optimisation routine for the MVR model.

        """
        # Create a likelihood
        self.loglikelihood = PreyPredGrowthMVRVirReadLogLik(
            self._model, self._viral_read_counts, self._vl_parameters,
            self._generation_times)

        # Determine total number of unique individuals sampled through
        # the viral read and Ct experiments
        self.total_times = self.loglikelihood.n_parameters()

        # Create a prior
        log_prior = mvri.MVRCtValLogPrior(self._model, self.total_times)

        # Create a posterior log-likelihood (log(likelihood * prior))
        self._log_posterior = pints.LogPosterior(self.loglikelihood, log_prior)
