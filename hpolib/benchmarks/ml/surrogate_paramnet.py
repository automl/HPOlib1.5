import os
import pickle
import numpy as np
import ConfigSpace as CS

from copy import deepcopy

from hpolib.abstract_benchmark import AbstractBenchmark


class SurrogateParamNet(AbstractBenchmark):

    def __init__(self, dataset, path=None, rng=None):

        super(SurrogateParamNet, self).__init__()

        self.surrogate_objective = pickle.load(open(os.path.join(path, "rf_surrogate_paramnet_%s.pkl" % dataset), "rb"))
        self.surrogate_cost = pickle.load(open(os.path.join(path, "rf_cost_surrogate_paramnet_%s.pkl" % dataset), "rb"))
        self.n_epochs = 50
        self.dataset = dataset
        if rng is None:
            self.rng = np.random.RandomState()
        else:
            self.rng = rng

    @AbstractBenchmark._check_configuration
    @AbstractBenchmark._configuration_as_array
    def objective_function(self, x, step=None, **kwargs):
        x_ = deepcopy(x)
        x_[0] = 10 ** x_[0]
        x_[1] = 2 ** x_[1]
        x_[2] = 2 ** x_[2]
        x_[3] = 10 ** x_[3]
        lc = self.surrogate_objective.predict(x_[None, :])[0]
        c = self.surrogate_cost.predict(x_[None, :])[0]

        if step is None:
            y = lc[-1]
        else:
            y = lc[step]

        return {'function_value': y, "cost": c}

    @AbstractBenchmark._check_configuration
    @AbstractBenchmark._configuration_as_array
    def objective_function_test(self, x, **kwargs):
        x_ = deepcopy(x)
        x_[0] = 10 ** x_[0]
        x_[1] = 2 ** x_[1]
        x_[2] = 2 ** x_[2]
        x_[3] = 10 ** x_[3]
        lc = self.surrogate_objective.predict(x_[None, :])[0]
        c = self.surrogate_cost.predict(x_[None, :])[0]
        y = lc[-1]
        return {'function_value': y, "cost": c}

    @staticmethod
    def get_configuration_space():
        cs = CS.ConfigurationSpace()
        cs.generate_all_continuous_from_bounds(SurrogateParamNet.get_meta_information()['bounds'])
        return cs

    @staticmethod
    def get_meta_information():
        return {'name': 'Stefans reparameterization of paramnet',
                # 'bounds': [[1e-6, 1e-2],  # initial_lr
                #            [8, 256],  # batch_size
                #            [16, 256],  # average_units_per_layer
                #            [1e-4, 1],  # final_lr_fraction
                #            [0, 1.],  # shape_parameter_1
                #            [1, 5],  # num_layers
                #            [0, .5],  # dropout_0
                #            [0, .5]]  # dropout_1
                'bounds': [[-6, -2],  # log10 initial_lr
                           [3, 8],  # log2 batch_size
                           [4, 8],  # log2 average_units_per_layer
                           [-4, 0],  # log10 final_lr_fraction
                           [0, 1.],  # shape_parameter_1
                           [1, 5],  # num_layers
                           [0, .5],  # dropout_0
                           [0, .5]]  # dropout_1
                }
