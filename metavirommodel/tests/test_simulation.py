#
# This file is part of metavirommodel (https://github.com/SABS-R3-Epidemiology/metavirommodel/)
# which is released under the BSD 3-clause license. See accompanying LICENSE.md
# for copyright notice and full license details.
#

import numpy as np

import unittest
import metavirommodel as vm


class TestSimulationController(unittest.TestCase):
    """
    Test the 'SimulationController' class.
    """
    def test__init__(self):

        start = 0
        end = 10
        with self.assertRaises(TypeError):
            vm.SimulationController(vm.SimulationController, start, end)
            vm.SimulationController('1', start, end)

    def test_run(self):

        start = 0
        end = 10
        model = vm.metavirommodeldel
        simulation = vm.SimulationController(model, start, end)

        initial_values = [0.9, 0, 0.1, 0]
        constants = [1, 1, 1]
        test_parameters = initial_values + constants
        output = simulation.run(test_parameters, ['S', 'E', 'I', 'R'])

        # Check output shape
        self.assertEqual(output.shape, (10, 4))

        # Check that sum of states is one at all times
        output = simulation.run(test_parameters)
        total = np.sum(output, axis=1)
        expected = np.ones(shape=10)
        np.testing.assert_almost_equal(total, expected)


if __name__ == '__main__':
    unittest.main()
