import numpy as np

import ConfigSpace as CS
from hpolib.abstract_benchmark import AbstractBenchmark


class CountingOnes(AbstractBenchmark):

    @AbstractBenchmark._check_configuration
    def objective_function(self, config, budget=100, **kwargs):

        y = 0
        for h in config:
            if type(h) == float:
                y += np.mean(np.random.binomial(1, config[h], budget))
            else:
                y += config[h]

        return {'function_value': -y}

    def objective_function_test(self, x, **kwargs):
        return self.objective_function(x)

    @staticmethod
    def get_configuration_space(n_categorical=1, n_continuous=1):
        cs = CS.ConfigurationSpace()
        for i in range(n_categorical):
            cs.add_hyperparameter(CS.CategoricalHyperparameter("cat_%d" % i, [0, 1]))
        for i in range(n_continuous):
            cs.add_hyperparameter(CS.UniformFloatHyperparameter('float_%d' % i, lower=0, upper=1))
        return cs

    @staticmethod
    def get_meta_information():
        return {'name': 'Counting Ones'}
