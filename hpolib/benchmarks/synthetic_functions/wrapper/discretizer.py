import numpy as np

from ConfigSpace import configuration_space, UniformIntegerHyperparameter, UniformFloatHyperparameter, CategoricalHyperparameter
from hpolib.abstract_wrapper import AbstractBenchmarkWrapper


class IntDiscretizeDimensions(AbstractBenchmarkWrapper):

    def __init__(self, original_benchmark, parameter, steps, rng=None):
        """
        Wrapper to discretize some dimensions of the searchspace. Discretized dimensions are handles as categoricals.

        **Note** Benchmark must not have conditionals/forbiddens

        :param original_benchmark:
        :param rng:
        :param parameter: list of strings of which parameters to discretize
        :param steps: list of integers with number of discretization steps to create
        """
        super().__init__(original_benchmark=original_benchmark, rng=rng)

        # Reset configuration space for original benchmark to make check_array not fail
        self.configuration_space = self.original_benchmark.get_configuration_space()

        self.parameter = parameter
        self.steps = steps
        self.stepsizes = {}
        self.lowers = {}

    def get_wrapper_label(self):
        return "int_discrete"

    def _transform_config_to_disc_space(self, x):
        if isinstance(x, np.ndarray) or isinstance(x, list):
            x = dict([("x%d" % i, v) for i, v in enumerate(x)])
        else:
            x = dict(x)

        # Check whether x is valid
        configuration_space.Configuration(self.get_configuration_space(), x)

        for name in self.parameter:
            x[name] = self.lowers[name] + self.stepsizes[name]*x[name]
        x = configuration_space.Configuration(self.original_benchmark.get_configuration_space(), x)
        return x

    def _transform_config_to_cont_space(self, x):
        if isinstance(x, np.ndarray) or isinstance(x, list):
            x = dict([("x%d" % i, v) for i, v in enumerate(x)])
        else:
            x = dict(x)

        # Check whether x is valid
        configuration_space.Configuration(self.configuration_space, x)

        for name in self.parameter:
            x[name] = int((x[name] - self.lowers[name]) / self.stepsizes[name])
        x = configuration_space.Configuration(self.get_configuration_space(), x)
        return x

    def get_configuration_space(self):
        cs = self.original_benchmark.get_configuration_space()
        new_cs = configuration_space.ConfigurationSpace()
        for hyper in cs.get_hyperparameters():
            if hyper.name in self.parameter:
                if type(hyper) == UniformFloatHyperparameter:
                    lower = hyper.lower
                    upper = hyper.upper
                    self.stepsizes[hyper.name] = (upper - lower) / (self.steps - 1)
                    self.lowers[hyper.name] = lower
                    p = UniformIntegerHyperparameter(hyper.name, 0, self.steps-1)
                    new_cs.add_hyperparameter(p)
                else:
                    raise ValueError("Can only discretize UniformFloatHyperparameter")
            else:
                new_cs.add_hyperparameter(hyper)
        return new_cs

    def objective_function(self, x, **kwargs):
        x = self._transform_config_to_disc_space(x)
        res = self.original_benchmark.objective_function(x, **kwargs)
        return res

    def objective_function_test(self, x, **kwargs):
        x = self._transform_config_to_disc_space(x)
        res = self.original_benchmark.objective_function_test(x, **kwargs)
        return res

    def get_meta_information(self):
        d = super().get_meta_information()
        del d["optima"]
        del d["f_opt"]
        return d


class CatDiscretizeDimensions(IntDiscretizeDimensions):

    def __init__(self, original_benchmark, parameter, steps, rng=None):
        """
        Wrapper to discretize some dimensions of the searchspace. Discretized dimensions are handles as categoricals.

        **Note** Benchmark must not have conditionals/forbiddens

        :param original_benchmark:
        :param rng:
        :param parameter: list of strings of which parameters to discretize
        :param steps: list of integers with number of discretization steps to create
        """
        super().__init__(original_benchmark=original_benchmark, parameter=parameter, steps=steps, rng=rng)

        # Reset configuration space for original benchmark to make check_array not fail
        self.configuration_space = self.original_benchmark.get_configuration_space()

        self.parameter = parameter
        self.steps = steps
        self.stepsizes = {}
        self.lowers = {}

    def get_wrapper_label(self):
        return "cat_discrete"

    def _transform_config_to_disc_space(self, x):
        if isinstance(x, np.ndarray) or isinstance(x, list):
            x = dict([("x%d" % i, v) for i, v in enumerate(x)])
        else:
            x = dict(x)

        # Check whether x is valid
        configuration_space.Configuration(self.get_configuration_space(), x)

        for name in self.parameter:
            x[name] = self.lowers[name] + self.stepsizes[name]*int(x[name])
        x = configuration_space.Configuration(self.original_benchmark.get_configuration_space(), x)
        return x

    def _transform_config_to_cont_space(self, x):
        if isinstance(x, np.ndarray) or isinstance(x, list):
            x = dict([("x%d" % i, v) for i, v in enumerate(x)])
        else:
            x = dict(x)

        # Check whether x is valid
        configuration_space.Configuration(self.configuration_space, x)

        for name in self.parameter:
            x[name] = int((x[name] - self.lowers[name]) / self.stepsizes[name])
        x = configuration_space.Configuration(self.get_configuration_space(), x)
        return x

    def get_configuration_space(self):
        cs = self.original_benchmark.get_configuration_space()
        new_cs = configuration_space.ConfigurationSpace()
        for hyper in cs.get_hyperparameters():
            if hyper.name in self.parameter:
                if type(hyper) == UniformFloatHyperparameter:
                    lower = hyper.lower
                    upper = hyper.upper
                    self.stepsizes[hyper.name] = (upper - lower) / (self.steps - 1)
                    self.lowers[hyper.name] = lower
                    p = CategoricalHyperparameter(hyper.name,
                                                  choices=[v for v in np.arange(start=0, stop=self.steps, step=1)])
                    new_cs.add_hyperparameter(p)
                else:
                    raise ValueError("Can only discretize UniformFloatHyperparameter")
            else:
                new_cs.add_hyperparameter(hyper)
        return new_cs
