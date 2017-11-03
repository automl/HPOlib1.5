import os
import pickle
import time
import numpy as np
import ConfigSpace as CS

from copy import deepcopy

from hpolib.abstract_benchmark import AbstractBenchmark


class Surrogate2DWideResNet(AbstractBenchmark):

    def __init__(self, path=None, rng=None):

        super(Surrogate2DWideResNet, self).__init__()

        self.surrogate = pickle.load(open(os.path.join(path, "rf_surrogate_2d_wide_res_net.pkl"), "rb"))

        if rng is None:
            self.rng = np.random.RandomState()
        else:
            self.rng = rng

    @AbstractBenchmark._check_configuration
    @AbstractBenchmark._configuration_as_array
    def objective_function(self, x, budget=4000, **kwargs):

        x_ = np.append(x, budget)[None, :]
        y = self.surrogate.predict(x_)[0]

        return {'function_value': (1 - y / 100.), "cost": budget}

    @AbstractBenchmark._check_configuration
    @AbstractBenchmark._configuration_as_array
    def objective_function_test(self, x, **kwargs):
        budget = 4000
        x_ = np.append(x, budget)[None, :]
        y = self.surrogate.predict(x_)[0]

        return {'function_value': (1 - y / 100.), "cost": budget}

    @staticmethod
    def get_configuration_space():
        cs = CS.ConfigurationSpace()
        cs.generate_all_continuous_from_bounds(Surrogate2DWideResNet.get_meta_information()['bounds'])
        return cs

    @staticmethod
    def get_meta_information():
        return {'name': 'Wide-ResNet with cosine annealing of the learning rate and a fix width of 32',
                'bounds': [[-2, 0],  # log10 initial_lr
                           [8, 26]]  # depth
                }
