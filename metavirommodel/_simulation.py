#
# This file is part of metavirommodel (https://github.com/SABS-R3-Epidemiology/metavirommodel/)
# which is released under the BSD 3-clause license. See accompanying LICENSE.md
# for copyright notice and full license details.
#

import numpy as np

import metavirommodel as se


class SimulationController(object):
    """SimulationController Class:

    Runs the simulation of any model and controls outputs

    Parameters
    ----------
    model: metavirommodel.ForwardModel class
    start: simulation start time
    end: simulation end time
    """

    def __init__(self, model, start, end): # noqa
        super(SimulationController, self).__init__()

        if not issubclass(model, se.ForwardModel):
            raise TypeError(
                'Model has to be a subclass of metavirommodel.ForwardModel.')

        self._model = model()
        self._simulation_times = np.arange(start, end, step=1)

    def run(self, parameters, outputs=None):

        # If outputs=None, the outputs will be the default values
        if outputs is not None:
            self._model.set_outputs(outputs)

        output = self._model.simulate(
            parameters,
            self._simulation_times)

        return output
