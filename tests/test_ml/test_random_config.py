import importlib
import inspect
import pkgutil
import os
import sys

import unittest
import unittest.mock

import numpy as np

import pynisher

import hpolib
import hpolib.benchmarks.ml
import hpolib.benchmarks.synthetic_functions
from hpolib.abstract_benchmark import AbstractBenchmark

os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,floatX=float32"


class TestRandomConfig(unittest.TestCase):

    def test_random_config_synthetic(self):
        path = os.path.dirname(hpolib.benchmarks.synthetic_functions.__file__)

        for _, pkg, _ in pkgutil.iter_modules([path, ]):
            pkg_name = "hpolib.benchmarks.synthetic_functions.{:s}".format(pkg)
            importlib.import_module(pkg_name)
            mod_name = sys.modules[pkg_name]

            for name, obj in inspect.getmembers(mod_name, inspect.isclass):
                if issubclass(obj, AbstractBenchmark) and "Abstract" not in name:
                    b = getattr(mod_name, name)()
                    cfg = b.get_configuration_space()
                    for i in range(100):
                        c = cfg.sample_configuration()
                        res = b.objective_function(c)
                        self.assertTrue(np.isfinite(res['function_value']))

    def test_random_config_ml(self):
        path = os.path.dirname(hpolib.benchmarks.ml.__file__)

        for _, pkg, _ in pkgutil.iter_modules([path, ]):
            pkg_name = "hpolib.benchmarks.ml.{:s}".format(pkg)
            importlib.import_module(pkg_name)
            mod_name = sys.modules[pkg_name]

            # Find Abstract Benchmark
            abstract_class = None
            for name, obj in inspect.getmembers(mod_name,
                                                inspect.isclass):
                if issubclass(obj, AbstractBenchmark) and \
                        inspect.isclass(obj) and \
                        AbstractBenchmark in obj.__bases__:
                    abstract_class = obj
                    break

            if abstract_class is not None:
                for name, obj in inspect.getmembers(mod_name, inspect.isclass):
                    if issubclass(obj, abstract_class) and \
                            inspect.isclass(obj) and \
                            abstract_class in obj.__bases__:
                        # Make sure to only test correct implementations
                        #print(mod_name, name, abstract_class)
                        b = getattr(mod_name, name)()
                        cfg = b.get_configuration_space()
                        for i in range(5):
                            c = cfg.sample_configuration()

                            # Limit Wallclocktime using pynisher
                            obj = pynisher.enforce_limits(wall_time_in_s=30)(b.objective_function)
                            res = obj(c)
                            if res is not None:
                                self.assertTrue(np.isfinite(res['cost']))
                                self.assertTrue(np.isfinite(res['function_value']))
                            else:
                                self.assertEqual(obj.exit_status,
                                                 pynisher.TimeoutException)
            else:
                raise ValueError("{:s} does not contain basic benchmark that is"
                                 " derived from AbstractBenchmark".
                                 format(mod_name))
