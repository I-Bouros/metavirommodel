#
# This file is part of metavirommodel
# (https://github.com/I-Bouros/metavirommodel)
# which is released under the BSD 3-clause license. See accompanying LICENSE.md
# for copyright notice and full license details.
#

import unittest

import numpy as np
import pandas as pd
import math
from scipy.stats import multinomial, gumbel_r

import metavirommodel as mvr
import metavirommodel.inference as mvri


#
# Toy Metaviromodel Model Class
#

class TestMetaviromodel(mvr.Metaviromodel):
    """
    Toy Metaviromodel model class used for testing.
    """
    def __init__(self):
        # Instantiate model
        super(TestMetaviromodel, self).__init__()

        self.algorithm = mvr.Metaviromodel()

        # Set initial reproduction number
        R_0 = 1

        # Set initial population state S - I - R
        N_init = 400
        # S_init = int(N_init / R_0)
        S_init = 380
        I_init = N_init - S_init
        R_init = 0
        initial_population = [S_init, I_init, R_init]

        self.init_cond = initial_population

        # Set birth rate
        theta = 0

        # Set death rates
        mu = 0
        nu = 0

        # Set transition rates
        infect_period = 15
        beta = R_0 / infect_period
        gamma = 1 / infect_period

        # Coalesce into paramater vector
        parameters = initial_population
        parameters.extend([theta, mu, nu, beta, gamma])

        self._set_parameters(parameters)
        self.parameters = parameters

    def __call__(self, total_days):
        # Select start and end times
        start_time = 1
        end_time = total_days

        return self.algorithm.simulate_fixed_times(
            self.parameters, start_time, end_time)


#
# Toy Viral Read Model Class
#

class TestViralReadModel(object):
    """
    Toy Viral Read Model class used for testing.
    """
    def __init__(self, model):
        # Set parameter for the viral read counts model
        self.t_eclipse = 3
        self.t_peak = 7
        self.t_switch = 5
        self.t_mod = 15
        self.t_LOD = math.inf

        self.sigma_obs = 0.25
        self.s_mod = 0.4
        self.v_zero = 2
        self.v_peak = 3880
        self.v_switch = 480
        self.v_LOD = 2

        self.parameters_vl = [
            self.t_eclipse, self.t_peak, self.t_switch, self.t_mod, self.t_LOD,
            self.v_zero, self.v_peak, self.v_switch, self.v_LOD,
            self.s_mod, self.sigma_obs]

        # Set read counts value for the suceptible and recovered individuals
        self.VR_susc = 0
        self.VR_rec = 18

        self.algorithm = model

        # Probability of clearing the virus
        self.p_addl = 0.2

    def compute_generation_times(self):
        generation_times = []

        for _ in range(70):
            if _ < self.t_eclipse + self.t_peak + self.t_switch:
                generation_times.append(
                    1-gumbel_r.cdf(
                        np.log(self.v_LOD),
                        self.algorithm._compute_mode_vr_model(
                            _, self.t_eclipse, self.t_peak, self.t_switch,
                            self.t_LOD, np.log(self.v_zero),
                            np.log(self.v_peak), np.log(self.v_switch),
                            np.log(self.v_LOD)),
                        self.algorithm._compute_sigma_vr_model(
                            _, self.t_eclipse, self.t_peak, self.t_switch,
                            self.t_mod, self.s_mod, self.sigma_obs)
                    ))

            else:
                generation_times.append(
                    (1-gumbel_r.cdf(
                        np.log(self.v_LOD),
                        self.algorithm._compute_mode_vr_model(
                            _, self.t_eclipse, self.t_peak, self.t_switch,
                            self.t_LOD, np.log(self.v_zero),
                            np.log(self.v_peak), np.log(self.v_switch),
                            np.log(self.v_LOD)),
                        self.algorithm._compute_sigma_vr_model(
                            _, self.t_eclipse, self.t_peak, self.t_switch,
                            self.t_mod, self.s_mod, self.sigma_obs)
                    )) * (1-self.p_addl)**(_ - self.t_eclipse - self.t_peak -
                                           self.t_switch))

        return generation_times

    def compute_R_history_clear(self, times, R_history, R_times_history):
        R_history_clear = []

        # Go through each recorded day
        for t, time in enumerate(times):
            current_clear_status = []

            # If there are any recovered individual
            if len(R_times_history[t]) > 0:
                # Go through each of them and
                for ind, ind_ID in enumerate(R_history[t]):
                    clear_status = 0

                    # If they have previously cleared the virus they signal
                    # that
                    if ind_ID in R_history[t-1] and (R_history_clear[-1][
                            R_history[t-1].index(ind_ID)] == 1):
                        clear_status = 1
                    # if not, they could do it today, if their time since
                    # infection exceeds teclipse + tpeak + tswitch
                    elif time > R_times_history[t][ind] + (self.t_eclipse +
                                                           self.t_peak +
                                                           self.t_switch):
                        clear_status = 1 - np.random.binomial(1, p=(
                            1-self.p_addl)**(time - R_times_history[t][ind] -
                                             self.t_eclipse - self.t_peak -
                                             self.t_switch))

                    current_clear_status.append(clear_status)

            R_history_clear.append(current_clear_status)

        return R_history_clear


#
# Toy Viral Read Data Class
#

class TestViralReadData(object):
    """
    Toy Viral Read Data class used for testing.
    """
    def __init__(self, model, viral_model):
        # Run Metaviromodel model
        self.model = model
        self.viral_model = viral_model

    def __call__(self, total_days):
        (output, S_history, I_history, R_history,
         I_times_history, R_times_history) = self.model(total_days)

        times = list(range(1, total_days+1))

        R_history_clear = self.viral_model.compute_R_history_clear(
            times, R_history, R_times_history)

        sample_points = np.arange(20, total_days, 30)
        sample_size = 20

        # Toy values for data structures about death
        vr_values = []
        vr_infec = []

        vr_susc_ids = []
        vr_infec_ids = []
        vr_recov_ids = []

        vr_time_of_recov_infec = []
        vr_time_of_infec = []
        vr_time_since_infec = []

        # At each point in time sample sample_size individuals
        for time in sample_points:
            # Identify the current infections at the specified timepoint
            current_susceptibles = S_history[time-1]
            current_infections = I_history[time-1]
            current_recovered = R_history[time-1]
            current_infection_times = I_times_history[time-1]
            current_recov_infection_times = R_times_history[time-1]
            current_recov_clear_virus_status = R_history_clear[time-1]

            # Sample without replacement the sample_size individuals and
            # determine their time since infection to produce Ct values
            # determine how many of those sampled are S, I and R
            number_selected_susc, number_selected_infec, number_selected_rec =\
                multinomial.rvs(
                    n=sample_size,
                    p=output[time-1, :]/np.sum(output[time-1, :]))

            # First add the Ct values for the sampled susceptibele and
            # recovered individuals
            sampled_vr_values = [
                self.viral_model.VR_susc] * number_selected_susc

            # determine the ids of those sampled Ss
            selected_individuals_susc_ids = np.random.choice(
                    current_susceptibles,
                    size=number_selected_susc,
                    replace=False).tolist()

            if len(current_recov_infection_times) > 0:
                # If we have at least one selected recovered
                # determine the indices of those sampled Rs
                selected_individuals_indices = np.random.choice(
                    range(len(current_recov_infection_times)),
                    size=number_selected_rec,
                    replace=False).tolist()

                selected_individuals_rec_ids = [
                    current_recovered[_] for _ in selected_individuals_indices]

                # Determine the time of infection of those sampled Rs
                selected_individuals_recov_infec_times = [
                    current_recov_infection_times[_] for _ in
                    selected_individuals_indices]

                # determine how long since infection for selected Rs
                sample_time_since_infec = time - \
                    selected_individuals_recov_infec_times

                # Determine the clearence of infection of those sampled Rs
                selected_individuals_clear_virus_status = [
                    current_recov_clear_virus_status[_] for _ in
                    selected_individuals_indices]

                # Run Ct model to determine individual Ct counts for each
                # sample
                for i, ti in enumerate(sample_time_since_infec):
                    sampled_vr_values.append(
                        self.model.algorithm.viral_read_model(
                            self.viral_model.parameters_vl, ti) *
                        selected_individuals_clear_virus_status[i])

            elif number_selected_rec > 0:
                # If initial step when no history of infection is provided
                for i in range(number_selected_rec):
                    sampled_vr_values.append(
                        self.model.algorithm.viral_read_model(
                            self.viral_model.parameters_vl, time))

                sample_time_since_infec = np.zeros(number_selected_rec)
                selected_individuals_rec_ids = []
            else:
                sample_time_since_infec = []
                selected_individuals_rec_ids = []

            if len(current_infection_times) > 0:
                # If we have at least one selected infection
                # determine the indices of those sampled Is
                selected_individuals_indices = np.random.choice(
                    range(len(current_infection_times)),
                    size=number_selected_infec,
                    replace=False).tolist()

                # Determine the ids of those sampled Is
                selected_individuals_infec_ids = [
                    current_infections[_] for _ in
                    selected_individuals_indices]

                # Determine the time of infection of those sampled Is
                selected_individuals_infec_times = [
                    current_infection_times[_] for _ in
                    selected_individuals_indices]

                # determine how long since infection for selected Is
                sample_time_since_infec = time - \
                    selected_individuals_infec_times

                # Run Ct model to determine individual Ct counts for each
                # sample
                for ti in sample_time_since_infec:
                    sampled_vr_values.append(
                        self.model.algorithm.viral_read_model(
                            self.viral_model.parameters_vl, ti))

            elif number_selected_infec > 0:
                # If initial step when no history of infection is provided
                for i in range(number_selected_infec):
                    sampled_vr_values.append(
                        self.model.algorithm.viral_read_model(
                            self.viral_model.parameters_vl, time))

                selected_individuals_infec_times = np.zeros(
                    number_selected_infec)
                sample_time_since_infec = np.zeros(number_selected_infec)
                selected_individuals_infec_ids = []
            else:
                selected_individuals_infec_times = []
                sample_time_since_infec = []
                selected_individuals_infec_ids = []

            vr_values.append(sampled_vr_values)
            vr_infec.append(number_selected_infec)

            vr_susc_ids.append(selected_individuals_susc_ids)
            vr_infec_ids.append(selected_individuals_infec_ids)
            vr_recov_ids.append(selected_individuals_rec_ids)

            vr_time_of_recov_infec.append(
                selected_individuals_recov_infec_times)
            vr_time_of_infec.append(selected_individuals_infec_times)
            vr_time_since_infec.append(sample_time_since_infec)

        vr_values = np.asarray(vr_values)
        vr_infec = np.asarray(vr_infec)

        vr_time_of_infec_data = pd.DataFrame(columns=['ID', 'Value'])
        for t, time in enumerate(sample_points):
            vr_time_of_infec_data = pd.concat(
                [
                    vr_time_of_infec_data,
                    pd.DataFrame({
                        'ID': vr_susc_ids[t] + vr_recov_ids[
                            t] + vr_infec_ids[t],
                        'Value': [400] * len(vr_susc_ids[
                            t]) + vr_time_of_recov_infec[
                                t] + vr_time_of_infec[t]
                    })
                ])

        vr_values_data = pd.DataFrame(columns=[
            'ID', 'TimeOfSample', 'Value'])
        for t, time in enumerate(sample_points):
            vr_values_data = pd.concat(
                [
                    vr_values_data,
                    pd.DataFrame({
                        'ID': vr_susc_ids[t] + vr_recov_ids[
                            t] + vr_infec_ids[t],
                        'TimeOfSample': [time] * sample_size,
                        'Value': vr_values[t, :].tolist()
                    })
                ])

        return vr_values_data


#
# Test MVRVirRead Log-Likelihood Class
#

class TestMVRVirReadLogLik(unittest.TestCase):
    """
    Test the 'MVRVirReadLogLik' class.
    """
    def test__call__(self):
        # Set times for inference
        total_days = 200

        # Set toy model, death and serology data
        model = TestMetaviromodel()
        viral_model = TestViralReadModel(model)

        generation_times = viral_model.compute_generation_times()
        vr_values_data = TestViralReadData(model, viral_model)(total_days)

        log_lik = mvri.MVRVirReadLogLik(
            model, vr_values_data, viral_model.parameters_vl, generation_times)

        self.assertIsInstance(log_lik([3, 0.002]), (int, float))
        self.assertEqual(log_lik([3, 0.002]) < 0, True)

    def test_n_parameters(self):
        # Set times for inference
        total_days = 200

        # Set toy model, death and serology data
        model = TestMetaviromodel()
        viral_model = TestViralReadModel(model)

        generation_times = viral_model.compute_generation_times()
        vr_values_data = TestViralReadData(model, viral_model)(total_days)

        log_lik = mvri.MVRVirReadLogLik(
            model, vr_values_data, viral_model.parameters_vl, generation_times)

        self.assertEqual(log_lik.n_parameters(), 2)


#
# Test MVRVirRead Inference and Optimisation Class
#

class TestMVRVirReadInfer(unittest.TestCase):
    """
    Test the 'MVRVirReadInfer' class.
    """
    def test__init__(self):
        # Set toy model, death and serology data
        model = TestMetaviromodel()
        viral_model = TestViralReadModel(model)

        generation_times = viral_model.compute_generation_times()

        # Set up MVRVirRead Inference class
        inference = mvri.MVRVirReadInfer(model, generation_times)

        self.assertIsInstance(inference._model, mvr.Metaviromodel)

        with self.assertRaises(TypeError):
            mvr.inference.MVRVirReadInfer(0)

    def test_read_viral_read_data(self):
        # Set times for inference
        total_days = 200

        # Set toy model, death and serology data
        model = TestMetaviromodel()
        viral_model = TestViralReadModel(model)

        generation_times = viral_model.compute_generation_times()
        vr_values_data = TestViralReadData(model, viral_model)(total_days)

        # Set up MVRVirRead Inference class
        inference = mvri.MVRVirReadInfer(model, generation_times)

        # Test read_viral_read_data
        inference.read_viral_read_data(vr_values_data,
                                       viral_model.parameters_vl)

        sample_points = np.arange(20, total_days, 30)
        sample_size = 20

        self.assertEqual(
            len(inference._viral_read_counts['ID'].values.tolist()),
            sample_size * sample_points.shape[0])
        self.assertEqual(
            len(inference._viral_read_counts['TimeOfSample'].values.tolist()),
            sample_size * sample_points.shape[0])
        self.assertEqual(
            len(inference._viral_read_counts['Value'].values.tolist()),
            sample_size * sample_points.shape[0])
        self.assertEqual(
            len(inference._vl_parameters), 11)

        self.assertEqual(inference._viral_read_counts['ID'].values.tolist(),
                         vr_values_data['ID'].values.tolist())
        self.assertEqual(inference._viral_read_counts[
            'TimeOfSample'].values.tolist(),
                         vr_values_data['TimeOfSample'].values.tolist())
        self.assertEqual(inference._viral_read_counts['Value'].values.tolist(),
                         vr_values_data['Value'].values.tolist())

        self.assertEqual(inference._vl_parameters, viral_model.parameters_vl)

    def test_return_loglikelihood(self):
        # Set times for inference
        total_days = 200

        # Set toy model, death and serology data
        model = TestMetaviromodel()
        viral_model = TestViralReadModel(model)

        generation_times = viral_model.compute_generation_times()
        vr_values_data = TestViralReadData(model, viral_model)(total_days)

        # Set up MVRVirRead Inference class
        inference = mvri.MVRVirReadInfer(model, generation_times)

        # Test read_viral_read_data
        inference.read_viral_read_data(vr_values_data,
                                       viral_model.parameters_vl)

        # Compute the log likelihood at chosen point in the parameter space
        log_lik = inference.return_loglikelihood([3, 0.002])

        self.assertIsInstance(log_lik, (int, float))
        self.assertEqual(log_lik < 0, True)

    def test_optimisation_problem_setup(self):
        # Set times for inference
        total_days = 200

        # Set toy model, death and serology data
        model = TestMetaviromodel()
        viral_model = TestViralReadModel(model)

        generation_times = viral_model.compute_generation_times()
        vr_values_data = TestViralReadData(model, viral_model)(total_days)

        # Set up MVRVirRead Inference class
        optimisation = mvri.MVRVirReadInfer(model, generation_times)

        # Test read_viral_read_data
        optimisation.read_viral_read_data(vr_values_data,
                                          viral_model.parameters_vl)

        # Set up and run the optimisation problem
        found, log_post_value = optimisation.optimisation_problem_setup()

        self.assertEqual(len(found), 2)
        self.assertIsInstance(log_post_value, (int, float))
        self.assertEqual(log_post_value < 0, True)

    def test_inference_problem_setup(self):
        # Set times for inference
        total_days = 200

        # Set toy model, death and serology data
        model = TestMetaviromodel()
        viral_model = TestViralReadModel(model)

        generation_times = viral_model.compute_generation_times()
        vr_values_data = TestViralReadData(model, viral_model)(total_days)

        # Set up MVRVirRead Inference class
        inference = mvri.MVRVirReadInfer(model, generation_times)

        # Test read_viral_read_data
        inference.read_viral_read_data(vr_values_data,
                                       viral_model.parameters_vl)
        # Set up and run the inference problem
        samples = inference.inference_problem_setup(num_iter=600)

        self.assertEqual(len(samples), 3)
        self.assertEqual(samples[0].shape, (600, 2))
