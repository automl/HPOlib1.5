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
from hpolib.benchmarks.ml.autosklearn_benchmark import AutoSklearnBenchmark, MulticlassClassificationBenchmark
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
                for name, obj in inspect.getmembers(mod_name, inspect.isclass):
                    if issubclass(obj, abstract_class) and \
                            inspect.isclass(obj) and \
                            abstract_class in obj.__bases__:
                        print(obj)
                        self.assertIn("rng", inspect.signature(getattr(mod_name, name)).parameters)


