import os
import pickle
import numpy as np
import ConfigSpace as CS

from hpolib.abstract_benchmark import AbstractBenchmark

from hpolib.util.download_surrogate import DownloadSurrogate
from hpolib.util.data_manager import SurrogateData


class SurrogateCNN(AbstractBenchmark):

    def __init__(self, path='./', rng=None):
        super(SurrogateCNN, self).__init__()

        url = "http://www.ml4aad.org/wp-content/uploads/2017/12/lcnet_datasets.zip"
        surrogate = SurrogateData(surrogate_file="surrogate_cnn.pkl", url=url, folder = "lcnet_datasets/convnet_cifar10/")
        self.surrogate_objective = surrogate.load_objective()
        self.surrogate_cost = surrogate.load_cost()
        self.n_epochs = 40
        if rng is None:
            self.rng = np.random.RandomState()
        else:
            self.rng = rng

    @AbstractBenchmark._check_configuration
    @AbstractBenchmark._configuration_as_array
    def objective_function(self, x, step=None, **kwargs):
        if step is None:
            step = self.n_epochs

        test_point = np.concatenate((x[None, :], np.array([[step]])), axis=1)

        y = self.surrogate_objective.predict(test_point)[0]

        c = np.cumsum([self.surrogate_cost.predict(np.concatenate((x[None, :], np.array([[i]])), axis=1))[0] for i in
                       range(1, step + 1)])[0]

        return {'function_value': y, "cost": c}

    @AbstractBenchmark._check_configuration
    @AbstractBenchmark._configuration_as_array
    def objective_function_test(self, x, **kwargs):
        return self.objective_function(x, step=self.n_epochs)

    @staticmethod
    def get_configuration_space():
        cs = CS.ConfigurationSpace()
        cs.generate_all_continuous_from_bounds(SurrogateCNN.get_meta_information()['bounds'])
        return cs

    @staticmethod
    def get_meta_information():
        return {'name': 'Convolutional Neural Network Surrogate',
                'bounds': [[0, 1],  # init_learning_rate
                           [0, 1],  # batch_size
                           [0, 1],  # n_units_1
                           [0, 1],  # n_units_2
                           [0, 1]],  # n_units_3
                }
