import numpy as np

from ConfigSpace import configuration_space, UniformIntegerHyperparameter, UniformFloatHyperparameter
from hpolib.abstract_wrapper import AbstractBenchmarkWrapper


class DiscretizeDimensions(AbstractBenchmarkWrapper):

    def __init__(self, original_benchmark, parameter, steps, rng=None):
        """
        Wrapper to discretize some dimensions of the searchspace.

        **Note** Benchmark must not have conditionals/forbiddens

        :param original_benchmark:
        :param rng:
        :param parameter: list of strings of which parameters to discretize
        :param steps: list of integers with number of discretization steps to create
        """
        super(DiscretizeDimensions, self).__init__(original_benchmark=original_benchmark, rng=rng)

        # Reset configuration space for original benchmark to make check_array not fail
        self.configuration_space = self.original_benchmark.get_configuration_space()

        self.parameter = parameter
        self.steps = steps
        self.stepsizes = {}
        self.lowers = {}

    def get_wrapper_label(self):
        return "discrete"

    def get_configuration_space(self):
        cs = self.original_benchmark.get_configuration_space()
        new_cs = configuration_space.ConfigurationSpace()
        for hyper in cs.get_hyperparameters():
            if hyper.name in self.parameter:
                if type(hyper) == UniformFloatHyperparameter:
                    lower = hyper.lower
                    upper = hyper.upper
                    self.stepsizes[hyper.name] = (upper - lower) / self.steps
                    self.lowers[hyper.name] = lower
                    p = UniformIntegerHyperparameter(hyper.name, 1, self.steps)
                    new_cs.add_hyperparameter(p)
                else:
                    raise ValueError("Can only discretize UniformFloatHyperparameter")
            else:
                new_cs.add_hyperparameter(hyper)
        return new_cs

    def objective_function(self, x, **kwargs):
        x = dict(x)
        print(x)
        print(self.configuration_space)
        print(self.get_configuration_space())
        for name in self.parameter:
            x[name] = self.lowers[name] + self.stepsizes[name]*x[name]
        x = configuration_space.Configuration(self.original_benchmark.get_configuration_space(), x)
        res = self.original_benchmark.objective_function(x, **kwargs)
        return res

    def objective_function_test(self, x, **kwargs):
        res = self.original_benchmark.objective_function_test(x, **kwargs)
        return res