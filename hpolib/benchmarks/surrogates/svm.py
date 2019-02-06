import numpy as np
import ConfigSpace as CS

from hpolib.abstract_benchmark import AbstractBenchmark
from hpolib.benchmarks.surrogates.surrogate_benchmark import SurrogateBenchmark


class SurrogateSVM(SurrogateBenchmark):

    def __init__(self, **kwargs):
        """

        Parameters
        ----------
        path: str
            directory to find or download dataset to
        """

        objective_fn = "rf_surrogate_svm.pkl"
        cost_fn = "rf_cost_surrogate_svm.pkl"

        super(SurrogateSVM, self).__init__(objective_surrogate_fn=objective_fn, cost_surrogate_fn=cost_fn, **kwargs)

        self.s_min = 100 / 50000.

    @AbstractBenchmark._check_configuration
    @AbstractBenchmark._configuration_as_array
    def objective_function(self, x, dataset_fraction=1, **kwargs):
        test_point = np.append(x, dataset_fraction)[None, :]

        y = self.surrogate_objective.predict(test_point)[0]
        c = self.surrogate_cost.predict(test_point)[0]
        return {'function_value': y, "cost": c}

    @AbstractBenchmark._check_configuration
    @AbstractBenchmark._configuration_as_array
    def objective_function_test(self, x, **kwargs):
        return self.objective_function(x, dataset_fraction=1, **kwargs)

    @staticmethod
    def get_configuration_space():
        cs = CS.ConfigurationSpace()
        cs.generate_all_continuous_from_bounds(SurrogateSVM.get_meta_information()['bounds'])
        return cs

    @staticmethod
    def get_meta_information():
        return {'name': 'Support Vector Machine',
                'bounds': [[-10, 10],   # C
                           [-10, 10]],  # gamma
                'references': ["@INPROCEEDINGS{klein-ejs17,"
                               "author = {A. Klein and S. Falkner and S. Bartels and P. Hennig and F. Hutter},"
                               "title = {Fast Bayesian hyperparameter optimization on large datasets},"
                               "booktitle = {Electronic Journal of Statistics},"
                               "year = {2017},"
                               "volume = {11}"]
                }
