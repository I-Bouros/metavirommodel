#
# This file is part of metavirommodel
# (https://github.com/I-Bouros/metavirommodel)
# which is released under the BSD 3-clause license. See accompanying LICENSE.md
# for copyright notice and full license details.
#
"""
This script contains code for indetifying optimal values for the non-inferred
transmission and viral/Ct model parameters based on the observed metarviromic
dataset provided.

It uses a latin hypercube sampling approach for testing possible fixed
parameter values.

"""

import math
import numpy as np
import pandas as pd
from scipy.stats import qmc, gumbel_r, poisson, gamma, beta

import metavirommodel as mvr
import metavirommodel.inference as mvri

from multiprocessing import Pool


#
# MVRHyperParameterSearch
#

class MVRHyperParameterSearch(object):
    """MVRVirReadLogLik Class:
    Controller class to construct the log-likelihood needed for optimisation or
    inference of the MVR model with constant growth in a PINTS framework.

    Parameters
    ----------
    model : Metaviromodel
        The model for which we solve the optimisation or inference problem.
    metaviromic_data : pandas.DataFrame
        Dataframe of the metaviromic data (either Viral read counts or Ct
        values), organised by individual ID and time of sample collection.
        Ordered by time of sample collection.
    inference_method : MVRVirReadInfer or MVRCtValInfer
        The inferenced method used for the model parameter fitting.
    parameters_margins : list of numpy.array
        List of parameters for the prior distributions on the parameters
        governing the viral read count model dynamics.

    """
    def __init__(self, model, metaviromic_data, inference_method,
                 parameter_margins):
        # Set the prerequisites for the latin hypercube inference wrapper
        # Model, Viral Read / Ct value data and Inference method
        if not isinstance(model, mvr.Metaviromodel):
            raise TypeError(
                'The population model must be of the Metaviromodel type.')
        if not isinstance(metaviromic_data, pd.DataFrame):
            raise TypeError(
                'The metaviromic dataset used must be stored in a '
                'Dataframe format.')
        if not issubclass(inference_method, (mvri.MVRVirReadInfer,
                                             mvri.MVRCtValInfer)):
            raise TypeError(
                'The inference method must be of the MVRVirReadInfer of '
                'MVRCtValInfer type.')

        self._model = model
        self._metaviromic_data = metaviromic_data
        self.inference_class = inference_method

        # List of all parameters that are targetted by the latin hypercube
        # method, i.e. parameters that are not inferred by the inference method
        self.parameter_list = [
            't_eclipse', 't_peak', 't_switch', 't_mod',
            'v_zero', 'v_peak', 'v_switch',
            's_mod', 'sigma_obs', 'p_addl']
        self.n_parameters = len(self.parameter_list)

        # Set margins for parameters of hypercube
        self._set_inverse_method(parameter_margins)

    def _inverse_sampling(self, untrans_parameters_sample):
        """
        Transforms the sampled values between 0 and 1 from the latin hypercube
        method into the correct trasnformed fixed method values, using the
        inverse sampling laws created at the initialisation of this object.

        Parameters
        ----------
        untrans_parameters_sample : numpy.array
            List of sampled values between 0 and 1 to be transformed.

        Returns
        -------
        transformed_parameters_sample : numpy.array
            List of transformed parameter values.

        """
        transformed_parameters_sample = []

        # t_eclipse
        transformed_parameters_sample.append(
            self.t_eclipse_inv_sampling.ppf(untrans_parameters_sample[0])[0])
        # t_peak
        transformed_parameters_sample.append(
            self.t_peak_inv_sampling.ppf(untrans_parameters_sample[1])[0])
        # t_switch
        transformed_parameters_sample.append(
            self.t_switch_inv_sampling.ppf(untrans_parameters_sample[2])[0])
        # t_mod
        transformed_parameters_sample.append(
            self.t_mod_inv_sampling.ppf(untrans_parameters_sample[3])[0])
        # t_LOD
        transformed_parameters_sample.append(math.inf)

        # v_zero / c_zero
        zero_val = self.v_c_zero_inv_sampling.ppf(untrans_parameters_sample[4])
        transformed_parameters_sample.append(zero_val)
        # v_peak / c_peak
        transformed_parameters_sample.append(
            self.v_c_peak_inv_sampling.ppf(untrans_parameters_sample[5]))
        # v_switch / c_switch
        transformed_parameters_sample.append(
            self.v_c_switch_inv_sampling.ppf(untrans_parameters_sample[6]))
        # v_LOD / c_LOD
        transformed_parameters_sample.append(zero_val)

        # s_mod
        transformed_parameters_sample.append(
            self.s_mode_inv_sampling.ppf(untrans_parameters_sample[7]))
        # sigma_obs
        transformed_parameters_sample.append(
            self.sigma_obs_inv_sampling.ppf(untrans_parameters_sample[8]))

        # p_addl
        transformed_parameters_sample.append(
            self.p_addl_inv_sampling.ppf(untrans_parameters_sample[9]))

        return transformed_parameters_sample

    def _set_inverse_method(self, parameter_margins):
        """
        Creates the inverse sampling functions used to transform draws from
        the latin hypercube into values corresponding to the fixed parameters
        of the Metaviromodel type model and inference framework

        Parameters
        ----------
        parameters_margins : list of numpy.array
            List of parameters for the prior distributions on the parameters
            governing the viral read count model dynamics.

        """
        # t_eclipse
        self.t_eclipse_inv_sampling = poisson(parameter_margins[0])
        # t_peak
        self.t_peak_inv_sampling = poisson(parameter_margins[1])
        # t_switch
        self.t_switch_inv_sampling = poisson(parameter_margins[2])
        # t_mod
        self.t_mod_inv_sampling = poisson(parameter_margins[3])

        # v_zero / c_zero
        self.v_c_zero_inv_sampling = gamma(
            parameter_margins[4][0], scale=1/parameter_margins[4][1])
        # v_peak / c_peak
        self.v_c_peak_inv_sampling = gamma(
            parameter_margins[5][0], scale=1/parameter_margins[5][1])
        # v_switch / c_switch
        self.v_c_switch_inv_sampling = gamma(
            parameter_margins[6][0], scale=1/parameter_margins[6][1])

        # s_mod
        self.s_mode_inv_sampling = gamma(
            parameter_margins[7][0], scale=1/parameter_margins[7][1])
        # sigma_obs
        self.sigma_obs_inv_sampling = gamma(
            parameter_margins[8][0], scale=1/parameter_margins[8][1])

        # p_addl
        self.p_addl_inv_sampling = beta(
            parameter_margins[9][0], 1/parameter_margins[9][1])

    def _create_generation_times(self, parameters_metaviromic, p_addl):
        """
        Creates the generation times interval using the current guess for the
        fixed method parameters.

        Parameters
        ----------
        parameters_metaviromic : list of numpy.array
            List of parameters governing the viral read count or Ct value model
            dynamics.
        p_addl : float or int
            Daily probability of recovered fully clearing the virus.

        Returns
        -------
        generation_times : numpy.array or list
            List of probabilities of observing a detercatble Viral read count
            or Ct value t days after infection.

        """
        generation_times = []

        ts = parameters_metaviromic[:5]
        vals = parameters_metaviromic[5:]

        # generation times if the model suports viral read counts data
        if hasattr(self._model, '_compute_mode_vr_model'
                   ) and callable(
                           self._model._compute_mode_vr_model):
            for _ in range(70):
                if _ < ts[0] + ts[1] + ts[2]:
                    generation_times.append(
                        1-gumbel_r.cdf(
                            np.log(vals[3]),
                            self._model._compute_mode_vr_model(
                                _, ts[0], ts[1], ts[2], ts[4],
                                np.log(vals[0]), np.log(vals[1]),
                                np.log(vals[2]), np.log(vals[3])),
                            self._model._compute_sigma_vr_model(
                                _, ts[0], ts[1], ts[2], ts[3],
                                vals[4], vals[5])
                        ))

                else:
                    generation_times.append(
                        (1-gumbel_r.cdf(
                            np.log(vals[3]),
                            self._model._compute_mode_vr_model(
                                _, ts[0], ts[1], ts[2], ts[4],
                                np.log(vals[0]), np.log(vals[1]),
                                np.log(vals[2]), np.log(vals[3])),
                            self._model._compute_sigma_vr_model(
                                _, ts[0], ts[1], ts[2], ts[3],
                                vals[4], vals[5])
                        )) * (1-p_addl)**(_ - ts[0] - ts[1] - ts[2]))

        # generation times if the model suports Ct values data
        elif hasattr(self._model, '_compute_mode_ct_model'
                     ) and callable(
                           self._model._compute_mode_ct_model):
            for _ in range(70):
                if _ < ts[0] + ts[1] + ts[2]:
                    generation_times.append(
                        gumbel_r.cdf(
                            vals[3],
                            self._model._compute_mode_ct_model(
                                _, ts[0], ts[1], ts[2], ts[4],
                                vals[0], vals[1], vals[2], vals[3]),
                            self._model._compute_sigma_ct_model(
                                _, ts[0], ts[1], ts[2], ts[3],
                                vals[4], vals[5])
                        ))

                else:
                    generation_times.append(
                        gumbel_r.cdf(
                            vals[3],
                            self._model._compute_mode_ct_model(
                                _, ts[0], ts[1], ts[2], ts[4],
                                vals[0], vals[1], vals[2], vals[3]),
                            self._model._compute_sigma_ct_model(
                                _, ts[0], ts[1], ts[2], ts[3],
                                vals[4], vals[5])
                        ) * (1-p_addl)**(_ - ts[0] - ts[1] - ts[2]))

        return generation_times

    def latin_hypercube_search(self, n_samples=100, n_cores=1):
        """
        Explores the multi-dimensional parameter hyperspace to identify
        choices of fixed method parameters that leads to largest
        log-likelihood values, after optimisation is completed.

        Parameters
        ----------
        n_samples : int
            Number of individuals draws from the multi-dimensional hypercube
            used for the latin hypercube sampling.
        n_cores : int
            Number of individual batches in which the number of samples from
            the multi-dimensional hypercube is partioned for running on
            individual cores.

        """
        # Sample 100 times from multidimensional hypercube
        untrans_parameters_samples = \
            qmc.LatinHypercube(self.n_parameters).random(n=n_samples)

        # Create dataframe in which we record all results
        results = pd.DataFrame(columns=[
            'Fixed Parameter Values', 'Inferred Parameter Values',
            'Log-Likelihood'])

        # Begin parallelisation process
        pool = Pool(n_cores)

        # Split samples into batches
        batches = np.linspace(0, n_samples, n_cores+1, dtype=int)

        blocks = []
        for i in range(len(batches)):
            if i < len(batches)-1:
                blocks.append(
                    untrans_parameters_samples[batches[i]:batches[i+1]])

        funclist = []
        for batch_untrans_parameters_samples in blocks:
            f = pool.apply_async(self._inference_method_parallelised,
                                 [batch_untrans_parameters_samples])
            funclist.append(f)

        results = []
        for f in funclist:
            batch_results = f.get(timeout=None)
            results.append(batch_results)
        pool.close()
        pool.join()

        print(results)

        results = pd.concat(results)

        # Return best result, as well as all pd.Dataframe of all results
        return (results.loc[
            results['Log-Likelihood'] == results['Log-Likelihood'].max()],
                results)

    def _inference_method_parallelised(self, untrans_parameters_samples):
        """
        Runs inference framework and return a dataframe of trasnformed fixed
        and inferred parameters and the log-likelihood associated with each
        of the draws from the latin hypercube method - parallised.

        Parameters
        ----------
        untrans_parameters_samples : list of numpy.array
            List of multi-dimensional hypercube draws used for the latin
            hypercube sampling.

        """
        # Create dataframe in which we record all results
        batch_results = pd.DataFrame(columns=[
            'Fixed Parameter Values', 'Inferred Parameter Values',
            'Log-Likelihood'])

        # For each sampled set of untransformed parameters
        for untrans_parameters_sample in untrans_parameters_samples:
            # Use inverse sampling method to retrieve
            # current guesses for fixed model parameters
            trans_parameter_guesses = self._inverse_sampling(
                untrans_parameters_sample)

            # Once this is done, we separted the parameters
            # into the distinct holders used for the model
            current_guess_parameters_metaviromic = trans_parameter_guesses[:-1]
            current_guess_p_addl = trans_parameter_guesses[-1]

            # Update model parameters based on current guesses
            self._model.gamma = \
                1 / np.sum(current_guess_parameters_metaviromic[:3])
            # and compute generation times
            self._generation_times = self._create_generation_times(
                current_guess_parameters_metaviromic, current_guess_p_addl
            )

            # Create inference controller object and read in the corresponding
            # metaviromic data
            self.inference_controller = self.inference_class(
                self._model, self._generation_times)

            if hasattr(self.inference_controller, 'read_viral_read_data'
                       ) and callable(
                           self.inference_controller.read_viral_read_data):
                self.inference_controller.read_viral_read_data(
                    self._metaviromic_data,
                    current_guess_parameters_metaviromic)
            elif hasattr(self.inference_controller, 'read_ct_values_data'
                         ) and callable(
                           self.inference_controller.read_ct_values_data):
                self.inference_controller.read_ct_values_data(
                    self._metaviromic_data,
                    current_guess_parameters_metaviromic)

            # Run optimiation algorithms
            found_ics, found_posterior_val = \
                self.inference_controller.optimisation_problem_setup()

            # Save results
            newrow = pd.DataFrame([{
                'Fixed Parameter Values': trans_parameter_guesses + [
                    self._model.gamma],
                'Inferred Parameter Values': found_ics,
                'Log-Likelihood': found_posterior_val
            }])

            print(newrow)
            batch_results = pd.concat([
                batch_results if not batch_results.empty else None,
                newrow])

        return batch_results
