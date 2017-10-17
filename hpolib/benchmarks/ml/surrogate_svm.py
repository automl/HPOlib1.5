import os
import pickle
import numpy as np
import ConfigSpace as CS

from hpolib.abstract_benchmark import AbstractBenchmark


class SurrogateSVM(AbstractBenchmark):

    def __init__(self, path=None, rng=None):
        """

        Parameters
        ----------
        path: str
            directory to find or download dataset to
        """

        super(SurrogateSVM, self).__init__()

        self.surrogate_objective = pickle.load(open(os.path.join(path, "rf_surrogate_svm.pkl"), "rb"))
        self.surrogate_cost = pickle.load(open(os.path.join(path, "rf_cost_surrogate_svm.pkl"), "rb"))

        self.s_min = 100 / 50000.

        if rng is None:
            self.rng = np.random.RandomState()
        else:
            self.rng = rng

    @AbstractBenchmark._check_configuration
    @AbstractBenchmark._configuration_as_array
    def objective_function(self, x, dataset_fraction=1, **kwargs):
        test_point = np.append(x, dataset_fraction)

        y = self.surrogate_objective.predict(test_point)
        c = self.surrogate_cost.predict(test_point)
        return {'function_value': y, "cost": c}

    @AbstractBenchmark._check_configuration
    @AbstractBenchmark._configuration_as_array
    def objective_function_test(self, x, **kwargs):
        test_point = np.append(x, 1)

        y = self.surrogate_objective.predict(test_point)
        c = self.surrogate_cost.predict(test_point)
        return {'function_value': y, "cost": c}

    @staticmethod
    def get_configuration_space():
        cs = CS.ConfigurationSpace()
        cs.generate_all_continuous_from_bounds(SurrogateSVM.get_meta_information()['bounds'])
        return cs

    @staticmethod
    def get_meta_information():
        return {'name': 'Support Vector Machine',
                'bounds': [[-10, 10],  # C
                           [-10, 10]],  # gamma
                'references': ["@article{klein-corr16,"
                               "author = {A. Klein and S. Falkner and S. Bartels and P. Hennig and F. Hutter},"
                               "title = {Fast Bayesian Optimization of Machine Learning"
                               "Hyperparameters on Large Datasets},"
                               "journal = corr,"
                               "llvolume = {abs/1605.07079},"
                               "lurl = {http://arxiv.org/abs/1605.07079}, year = {2016} }"]
                }
