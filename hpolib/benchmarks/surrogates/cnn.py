import numpy as np
import ConfigSpace as CS

from hpolib.abstract_benchmark import AbstractBenchmark
from hpolib.benchmarks.surrogates.surrogate_benchmark import SurrogateBenchmark


class SurrogateCNN(SurrogateBenchmark):

    def __init__(self, rng=None, path=None):

        self.n_epochs = 40

        objective_fn = "rf_surrogate_cnn.pkl"
        cost_fn = "rf_cost_surrogate_cnn.pkl"
        super(SurrogateCNN, self).__init__(objective_surrogate_fn=objective_fn, cost_surrogate_fn=cost_fn,
                                           path=path, rng=rng)

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
                'bounds': [[0, 1],  # init_learning_rate, [10**-6, 10**0]
                           [0, 1],  # batch_size, [32, 512]
                           [0, 1],  # n_units_1,  [2**4, 2**8]
                           [0, 1],  # n_units_2,  [2**4, 2**8]
                           [0, 1]],  # n_units_3, [2**4, 2**8]
                'references': ["@InProceedings{klein-iclr17,"
                               "author = {A. Klein and S. Falkner and J. T. Springenberg and F. Hutter},"
                               "title = {Learning Curve Prediction with {Bayesian} Neural Networks},"
                               "booktitle = {International Conference on Learning Representations (ICLR)"
                               " 2017 Conference Track},"
                               "year = {2017},"
                               "month = apr"]
                }
