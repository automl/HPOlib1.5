import numpy as np

from hpolib.abstract_wrapper import AbstractBenchmarkWrapper


class Log10ObjectiveValue(AbstractBenchmarkWrapper):

    def objective_function(self, x, **kwargs):
        res = self.original_benchmark.objective_function(x, **kwargs)
        res['function_value'] = np.log10(res['function_value'])
        return res

    def objective_function_test(self, x, **kwargs):
        res = self.original_benchmark.objective_function_test(x, **kwargs)
        res['function_value'] = np.log10(res['function_value'])
        return res


class ExpObjectiveValue(AbstractBenchmarkWrapper):

    def objective_function(self, x, **kwargs):
        res = self.original_benchmark.objective_function(x, **kwargs)
        res['function_value'] = np.exp(res['function_value'])
        return res

    def objective_function_test(self, x, **kwargs):
        res = self.original_benchmark.objective_function_test(x, **kwargs)
        res['function_value'] = np.exp(res['function_value'])
        return res
