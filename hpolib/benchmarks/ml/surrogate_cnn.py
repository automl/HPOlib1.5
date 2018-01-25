import os
import pickle
import numpy as np
import ConfigSpace as CS

from hpolib.abstract_benchmark import AbstractBenchmark

from hpolib.util.download_surrogate import DownloadSurrogate


class SurrogateCNN(AbstractBenchmark):

    def __init__(self, path=None, rng=None):
        super(SurrogateCNN, self).__init__()

        if path is None:
            url = "http://www.ml4aad.org/wp-content/uploads/2017/12/lcnet_datasets.zip"
            surrogate = DownloadSurrogate(url, folder = "lcnet_datasets/convnet_cifar10/")
            self.surrogate_objective = surrogate.get_learning_curve_rf()
            self.surrogate_cost = surrogate.get_time_cost_rf()
        else:
            self.surrogate_objective = pickle.load(open(os.path.join(path, "rf_surrogate_cnn.pkl"), "rb"))
            self.surrogate_cost = pickle.load(open(os.path.join(path, "rf_cost_surrogate_cnn.pkl"), "rb"))

        self.n_epochs = 40
        if rng is None:
            self.rng = np.random.RandomState()
        else:
            self.rng = rng

    @AbstractBenchmark._check_configuration
    @AbstractBenchmark._configuration_as_array
    def objective_function(self, x, step=None, **kwargs):
        if step is None:
            test_point = np.concatenate((x[None, :], np.array([[self.n_epochs]])), axis=1)
        else:
            test_point = np.concatenate((x[None, :], np.array([[step]])), axis=1)

        y = self.surrogate_objective.predict(test_point)[0]
        c = self.surrogate_cost.predict(test_point)[0]
        return {'function_value': y, "cost": c}

    @AbstractBenchmark._check_configuration
    @AbstractBenchmark._configuration_as_array
    def objective_function_test(self, x, **kwargs):
        test_point = np.concatenate((x[None, :], np.array([[self.n_epochs]])), axis=1)

        y = self.surrogate_objective.predict(test_point)[0]
        c = self.surrogate_cost.predict(test_point)[0]
        return {'function_value': y, "cost": c}

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
