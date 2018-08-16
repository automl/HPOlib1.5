import ConfigSpace as CS
from hpolib.abstract_benchmark import AbstractBenchmark


class Rosenbrock(AbstractBenchmark):
    @AbstractBenchmark._check_configuration
    @AbstractBenchmark._configuration_as_array
    def objective_function(self, x, **kwargs):
        y = 0
        d = 2
        for i in range(d - 1):
            y += 100 * (x[i + 1] - x[i] ** 2) ** 2
            y += (x[i] - 1) ** 2

        return {'function_value': y}

    def objective_function_test(self, x, **kwargs):
        return self.objective_function(x)

    @staticmethod
    def get_configuration_space():
        cs = CS.ConfigurationSpace()
        cs.generate_all_continuous_from_bounds(Rosenbrock.get_meta_information()['bounds'])
        return cs

    @staticmethod
    def get_meta_information():
        return {'name': 'Rosenbrock',
                'num_function_evals': 200,
                'optima': ([[1, 1]]),
                'bounds': [[-5.0, 10.0], [-5, 10.0]],
                'f_opt': 0.0}


class Rosenbrock5D(AbstractBenchmark):
    @AbstractBenchmark._check_configuration
    @AbstractBenchmark._configuration_as_array
    def objective_function(self, x, **kwargs):
        y = 0
        d = 5
        for i in range(d - 1):
            y += 100 * (x[i + 1] - x[i] ** 2) ** 2
            y += (x[i] - 1) ** 2

        return {'function_value': y}

    def objective_function_test(self, x, **kwargs):
        return self.objective_function(x)

    @staticmethod
    def get_configuration_space():
        cs = CS.ConfigurationSpace()
        cs.generate_all_continuous_from_bounds(Rosenbrock5D.get_meta_information()['bounds'])
        return cs

    @staticmethod
    def get_meta_information():
        return {'name': 'Rosenbrock5D',
                'num_function_evals': 500,
                'optima': ([[1] * 5]),
                'bounds': [[-5.0, 10.0]] * 5,
                'f_opt': 0.0}


class Rosenbrock10D(AbstractBenchmark):
    @AbstractBenchmark._check_configuration
    @AbstractBenchmark._configuration_as_array
    def objective_function(self, x, **kwargs):
        y = 0
        d = 10
        for i in range(d - 1):
            y += 100 * (x[i + 1] - x[i] ** 2) ** 2
            y += (x[i] - 1) ** 2

        return {'function_value': y}

    def objective_function_test(self, x, **kwargs):
        return self.objective_function(x)

    @staticmethod
    def get_configuration_space():
        cs = CS.ConfigurationSpace()
        cs.generate_all_continuous_from_bounds(Rosenbrock10D.get_meta_information()['bounds'])
        return cs

    @staticmethod
    def get_meta_information():
        return {'name': 'Rosenbrock10D',
                'num_function_evals': 1000,
                'optima': ([[1] * 10]),
                'bounds': [[-5.0, 10.0]] * 10,
                'f_opt': 0.0}


class Rosenbrock20D(AbstractBenchmark):
    @AbstractBenchmark._check_configuration
    @AbstractBenchmark._configuration_as_array
    def objective_function(self, x, **kwargs):
        y = 0
        d = 20
        for i in range(d - 1):
            y += 100 * (x[i + 1] - x[i] ** 2) ** 2
            y += (x[i] - 1) ** 2

        return {'function_value': y}

    def objective_function_test(self, x, **kwargs):
        return self.objective_function(x)

    @staticmethod
    def get_configuration_space():
        cs = CS.ConfigurationSpace()
        cs.generate_all_continuous_from_bounds(Rosenbrock20D.get_meta_information()['bounds'])
        return cs

    @staticmethod
    def get_meta_information():
        return {'name': 'Rosenbrock20D',
                'num_function_evals': 2000,
                'optima': ([[1] * 20]),
                'bounds': [[-5.0, 10.0]] * 20,
                'f_opt': 0.0}
