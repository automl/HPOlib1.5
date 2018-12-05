import abc

from hpolib.util import rng_helper
from hpolib.abstract_benchmark import AbstractBenchmark


class AbstractBenchmarkWrapper(object, metaclass=abc.ABCMeta):

    def __init__(self, original_benchmark, rng=None, **kwargs):
        """
        Interface for a wrapper to transform benchmarks

        :param original_benchmark: HPOlib2 benchmark
        :param rng: None, RandomState or int
        :param kwargs: any additional arguments required to initialize original benchmark
        """
        self.rng = rng_helper.create_rng(rng)
        self.original_benchmark = original_benchmark(rng=self.rng, **kwargs)

    def objective_function(self, x, **kwargs):
        """
        Returns objective function value for configuration x

        :param x: configuration
        :return: dict with at least key "function_value"
        """
        res = self.original_benchmark.objective_function(x, **kwargs)
        return res

    def objective_function_test(self, x, **kwargs):
        """
        Returns test function value for configuration x

        :param x: configuration
        :return: dict with at least key "function_value"
        """
        res = self.original_benchmark.objective_function_test(x, **kwargs)
        return res

    def get_configuration_space(self):
        """
        Returns config space, no longer static
        """
        return self.original_benchmark.get_configuration_space()

    def get_wrapper_label(self):
        return self.__class__.__name__

    def get_meta_information(self):
        """
        Returns meta information, no longer static
        """
        d = self.original_benchmark.get_meta_information()
        d['name'] = "%s(%s)" % (self.get_wrapper_label(), d['name'])
        return d
