#
# This file is part of metavirommodel
# (https://github.com/I-Bouros/metavirommodel)
# which is released under the BSD 3-clause license. See accompanying LICENSE.md
# for copyright notice and full license details.
#

import unittest

import numpy as np

import metavirommodel as mm


class TestPreyPredMetaviromodel(unittest.TestCase):
    """
    Test the 'PreyPredMetaviromodel' class.
    """
    def test__init__(self):
        model = mm.PreyPredMetaviromodel(20, 7, 2, 2, 10)
        self.assertEqual(model._output_names, ['S', 'I', 'R', 'P'])
        self.assertEqual(model._parameter_names, [
            'S0', 'I0', 'R0', 'P0', 'theta', 'mu', 'nu', 'beta', 'gamma',
            'pred_birth', 'pred_death_H', 'pred_death_L'
        ])
        self.assertEqual(model._n_outputs, 4)
        self.assertEqual(model._n_parameters, 12)
        np.testing.assert_array_equal(model._output_indices, np.arange(4))

        with self.assertRaises(TypeError):
            model = mm.PreyPredMetaviromodel(20, 7, 2, '2', 10)
        with self.assertRaises(TypeError):
            model = mm.PreyPredMetaviromodel(20, 7, 2, 2, '10')

    def test_n_outputs(self):
        model = mm.PreyPredMetaviromodel(20, 7, 2, 2, 10)
        self.assertEqual(model.n_outputs(), 4)

    def test_n_parameters(self):
        model = mm.PreyPredMetaviromodel(20, 7, 2, 2, 10)
        self.assertEqual(model.n_parameters(), 12)

    def test_output_names(self):
        model = mm.PreyPredMetaviromodel(20, 7, 2, 2, 10)
        self.assertEqual(model.output_names(), ['S', 'I', 'R', 'P'])

        model.set_outputs(['S', 'I'])
        self.assertEqual(model.output_names(), ['S', 'I'])

    def test_parameter_names(self):
        model = mm.PreyPredMetaviromodel(20, 7, 2, 2, 10)
        self.assertEqual(model.parameter_names(), [
            'S0', 'I0', 'R0', 'P0', 'theta', 'mu', 'nu', 'beta', 'gamma',
            'pred_birth', 'pred_death_H', 'pred_death_L'
        ])

    def test_set_outputs(self):
        model = mm.PreyPredMetaviromodel(20, 7, 2, 2, 10)

        # Check ValueError will be raised when some output names
        # are not as required
        with self.assertRaises(ValueError):
            model.set_outputs(['incidence number'])

        model.set_outputs(['I', 'R'])
        # Check the outputs names and number are as expected
        self.assertEqual(model._output_indices, [1, 2])
        self.assertEqual(model._n_outputs, 2)

    def test_simulate_fixed_times(self):
        model = mm.PreyPredMetaviromodel(20, 7, 2, 2, 10)

        initial_values = [10, 1, 1, 5]
        constants = [0.5, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.01, 0.1, 0.1]
        test_parameters = initial_values + constants

        model.set_outputs(['S', 'I', 'R', 'P'])
        (output, S_history, I_history, R_history,
         I_times_history, R_times_history) = \
            model.simulate_fixed_times(
                test_parameters, 1, 50)

        # Check output shape
        self.assertEqual(output.shape, (50, 4))
        self.assertEqual(len(S_history), 50)
        self.assertEqual(len(I_history), 50)
        self.assertEqual(len(R_history), 50)
        self.assertEqual(len(I_times_history), 50)
        self.assertEqual(len(R_times_history), 50)

        (output, S_history, I_history, R_history,
         I_times_history, R_times_history) = \
            model.simulate_fixed_times(
                [10, 1, 1, 5,
                 lambda _: 0.5, lambda _: 0.1, lambda _: 0.1,
                 0.2, 0.1, lambda _: 0.1, lambda _: 0.1, lambda _: 0.01,
                 lambda _: 0.1, lambda _: 0.1], 1, 50)

        # Check output shape
        self.assertEqual(output.shape, (50, 4))
        self.assertEqual(len(S_history), 50)
        self.assertEqual(len(I_history), 50)
        self.assertEqual(len(R_history), 50)
        self.assertEqual(len(I_times_history), 50)
        self.assertEqual(len(R_times_history), 50)

        # Check that simulation times meet conditions
        with self.assertRaises(TypeError):
            model.simulate_fixed_times(test_parameters, '1', 50)

        with self.assertRaises(ValueError):
            model.simulate_fixed_times(test_parameters, 0, 50)

        with self.assertRaises(TypeError):
            model.simulate_fixed_times(test_parameters, 1, '50')

        with self.assertRaises(ValueError):
            model.simulate_fixed_times(test_parameters, 1, 0)

        with self.assertRaises(ValueError):
            model.simulate_fixed_times(test_parameters, 100, 50)

        # Check that parameters meet conditions
        with self.assertRaises(TypeError):
            test_parameters1 = (10, 1, 1, 5, 0.5, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1,
                                0.01, 0.1, 0.1)
            model.simulate_fixed_times(test_parameters1, 1, 50)

        with self.assertRaises(ValueError):
            test_parameters1 = [10, 1, 1, 5, 0.5, 0.1, 0.1, 0.2, 0.1, 0.1]
            model.simulate_fixed_times(test_parameters1, 1, 50)

        with self.assertRaises(TypeError):
            test_parameters1 = [10, '1', 1, 5, 0.5, 0.1, 0.1, 0.2, 0.1, 0.1,
                                0.1, 0.01, 0.1, 0.1]
            model.simulate_fixed_times(test_parameters1, 1, 50)

        with self.assertRaises(TypeError):
            test_parameters1 = [10.0, 1, 1, 5, 0.5, 0.1, 0.1, 0.2, 0.1, 0.1,
                                0.1, 0.01, 0.1, 0.1]
            model.simulate_fixed_times(test_parameters1, 1, 50)

        with self.assertRaises(ValueError):
            test_parameters1 = [10, -1, 1, 5, 0.5, 0.1, 0.1, 0.2, 0.1, 0.1,
                                0.1, 0.01, 0.1, 0.1]
            model.simulate_fixed_times(test_parameters1, 1, 50)

        with self.assertRaises(TypeError):
            test_parameters1 = [10, 1, 1, 5, '0.5', 0.1, 0.1, 0.2, 0.1, 0.1,
                                0.1, 0.01, 0.1, 0.1]
            model.simulate_fixed_times(test_parameters1, 1, 50)

        with self.assertRaises(ValueError):
            test_parameters1 = [10, 1, 1, 5, -0.5, 0.1, 0.1, 0.2, 0.1, 0.1,
                                0.1, 0.01, 0.1, 0.1]
            model.simulate_fixed_times(test_parameters1, 1, 50)

        with self.assertRaises(TypeError):
            test_parameters1 = [10, 1, 1, 5, 0.5, '0.1', 0.1, 0.2, 0.1, 0.1,
                                0.1, 0.01, 0.1, 0.1]
            model.simulate_fixed_times(test_parameters1, 1, 50)

        with self.assertRaises(TypeError):
            test_parameters1 = [10, 1, 1, 5, 0.5, 0.1, '0.1', 0.2, 0.1, 0.1,
                                0.1, 0.01, 0.1, 0.1]
            model.simulate_fixed_times(test_parameters1, 1, 50)

        with self.assertRaises(ValueError):
            test_parameters1 = [10, 1, 1, 5, 0.5, -0.1, 0.1, 0.2, 0.1, 0.1,
                                0.1, 0.01, 0.1, 0.1]
            model.simulate_fixed_times(test_parameters1, 1, 50)

        with self.assertRaises(ValueError):
            test_parameters1 = [10, 1, 1, 5, 0.5, 0.1, -0.1, 0.2, 0.1, 0.1,
                                0.1, 0.01, 0.1, 0.1]
            model.simulate_fixed_times(test_parameters1, 1, 50)

        with self.assertRaises(TypeError):
            test_parameters1 = [10, 1, 1, 5, 0.5, 0.1, 0.1, '0.2', 0.1, 0.1,
                                0.1, 0.01, 0.1, 0.1]
            model.simulate_fixed_times(test_parameters1, 1, 50)

        with self.assertRaises(ValueError):
            test_parameters1 = [10, 1, 1, 5, 0.5, 0.1, 0.1, -0.2, 0.1, 0.1,
                                0.1, 0.01, 0.1, 0.1]
            model.simulate_fixed_times(test_parameters1, 1, 50)

        with self.assertRaises(TypeError):
            test_parameters1 = [10, 1, 1, 5, 0.5, 0.1, 0.1, 0.2, '0.1', 0.1,
                                0.1, 0.01, 0.1, 0.1]
            model.simulate_fixed_times(test_parameters1, 1, 50)

        with self.assertRaises(ValueError):
            test_parameters1 = [10, 1, 1, 5, 0.5, 0.1, 0.1, 0.2, -0.1, 0.1,
                                0.1, 0.01, 0.1, 0.1]
            model.simulate_fixed_times(test_parameters1, 1, 50)

        with self.assertRaises(TypeError):
            test_parameters1 = [10, 1, 1, 5, 0.5, 0.1, 0.1, 0.2, 0.1, '0.1',
                                0.1, 0.01, 0.1, 0.1]
            model.simulate_fixed_times(test_parameters1, 1, 50)

        with self.assertRaises(ValueError):
            test_parameters1 = [10, 1, 1, 5, 0.5, 0.1, 0.1, 0.2, 0.1, -0.1,
                                0.1, 0.01, 0.1, 0.1]
            model.simulate_fixed_times(test_parameters1, 1, 50)

        with self.assertRaises(TypeError):
            test_parameters1 = [10, 1, 1, 5, 0.5, 0.1, 0.1, 0.2, 0.1, 0.1,
                                '0.1', 0.01, 0.1, 0.1]
            model.simulate_fixed_times(test_parameters1, 1, 50)

        with self.assertRaises(ValueError):
            test_parameters1 = [10, 1, 1, 5, 0.5, 0.1, 0.1, 0.2, 0.1, 0.1,
                                -0.1, 0.01, 0.1, 0.1]
            model.simulate_fixed_times(test_parameters1, 1, 50)

        with self.assertRaises(TypeError):
            test_parameters1 = [10, 1, 1, 5, 0.5, 0.1, 0.1, 0.2, 0.1, 0.1,
                                0.1, '0.01', 0.1, 0.1]
            model.simulate_fixed_times(test_parameters1, 1, 50)

        with self.assertRaises(ValueError):
            test_parameters1 = [10, 1, 1, 5, 0.5, 0.1, 0.1, 0.2, 0.1, 0.1,
                                0.1, -0.01, 0.1, 0.1]
            model.simulate_fixed_times(test_parameters1, 1, 50)

        with self.assertRaises(TypeError):
            test_parameters1 = [10, 1, 1, 5, 0.5, 0.1, 0.1, 0.2, 0.1, 0.1,
                                0.1, 0.01, '0.1', 0.1]
            model.simulate_fixed_times(test_parameters1, 1, 50)

        with self.assertRaises(ValueError):
            test_parameters1 = [10, 1, 1, 5, 0.5, 0.1, 0.1, 0.2, 0.1, 0.1,
                                0.1, 0.01, -0.1, 0.1]
            model.simulate_fixed_times(test_parameters1, 1, 50)

        with self.assertRaises(TypeError):
            test_parameters1 = [10, 1, 1, 5, 0.5, 0.1, 0.1, 0.2, 0.1, 0.1,
                                0.1, 0.01, 0.1, '0.1']
            model.simulate_fixed_times(test_parameters1, 1, 50)

        with self.assertRaises(ValueError):
            test_parameters1 = [10, 1, 1, 5, 0.5, 0.1, 0.1, 0.2, 0.1, 0.1,
                                0.1, 0.01, 0.1, -0.1]
            model.simulate_fixed_times(test_parameters1, 1, 50)
