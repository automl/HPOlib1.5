import ConfigSpace as CS
import numpy as np

from hpolib.abstract_benchmark import AbstractBenchmark


class Rosenbrock(AbstractBenchmark):

    def __init__(self, d=2, **kwargs):
        self.d = d
        super(Rosenbrock, self).__init__(**kwargs)

    @AbstractBenchmark._check_configuration
    @AbstractBenchmark._configuration_as_array
    def objective_function(self, x, **kwargs):
        y = 0
        for i in range(self.d - 1):
            y += 100 * (x[i + 1] - x[i] ** 2) ** 2
            y += (x[i] - 1) ** 2

        return {'function_value': y}

    def objective_function_test(self, x, **kwargs):
        return self.objective_function(x)

    def get_configuration_space(self):
        cs = CS.ConfigurationSpace()
        cs.generate_all_continuous_from_bounds(self.get_meta_information()['bounds'])
        return cs

    def get_meta_information(self):
        return {'name': 'Rosenbrock',
                'num_function_evals': self.d * 100,
                'optima': ([[1] * self.d]),
                'bounds': [[-5, 10]] * self.d,
                'f_opt': 0.0}


class MultiFidelityRosenbrock(Rosenbrock):

    @AbstractBenchmark._check_configuration
    @AbstractBenchmark._configuration_as_array
    def objective_function(self, x, budget=100, **kwargs):
        #shift = (10 + 2 * np.log(budget / 100.)) / 10.

        #f_bias = np.log(100) - np.log(budget)
        f_bias = 0
        shift = 2 - 2 * (budget / 100)
        y = 0
        for i in range(self.d - 1):
            zi = x[i] - shift
            zi_next = x[i + 1] - shift

            y += 100 * (zi_next - zi ** 2) ** 2
            y += (zi - 1) ** 2
            y += f_bias

        return {'function_value': y}

    def objective_function_test(self, x, **kwargs):
        return self.objective_function(x, budget=100)
