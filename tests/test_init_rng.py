import importlib
import inspect
import pkgutil
import os
import sys

import unittest
import unittest.mock

import hpolib
#import hpolib.util.rng_helper
import hpolib.benchmarks.ml

from hpolib.abstract_benchmark import AbstractBenchmark

import numpy as np


class TestInitRng(unittest.TestCase):

    @unittest.mock.patch('hpolib.util.rng_helper.get_rng')
    def test_init_rng(self, rng_mock):
        rng_mock.side_effect = Exception("Used get_rng")
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

            # Check whether all classes accept rng
            if abstract_class is not None:
                for name, obj in inspect.getmembers(mod_name,
                                                inspect.isclass):
                    if issubclass(obj, abstract_class) and \
                            inspect.isclass(obj) and \
                            abstract_class in obj.__bases__:
                        print(obj)

                        rng = np.random.RandomState(1)
                        b = getattr(mod_name, name)(rng=rng)
                        self.assertListEqual(list(b.rng.get_state()[1]),
                                             list(rng.get_state()[1]))

                        b = getattr(mod_name, name)(rng=1)
                        rng = np.random.RandomState(1)
                        self.assertListEqual(list(b.rng.get_state()[1]),
                                             list(rng.get_state()[1]))

                        cs = b.get_configuration_space()
                        cfg = cs.sample_configuration()
                        self.assertRaisesRegex(Exception, "Used get_rng",
                                               b.objective_function,
                                               cfg, seed=1)

                        self.assertRaisesRegex(Exception, "Used get_rng",
                                               b.objective_function_test,
                                               cfg, seed=1)



