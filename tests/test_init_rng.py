import importlib
import inspect
import pkgutil
import os
import sys

import unittest
import unittest.mock

import hpolib
import hpolib.benchmarks.ml
from hpolib.abstract_benchmark import AbstractBenchmark

import numpy as np


class TestInitRng(unittest.TestCase):

    def test_init_rng(self):
        path = os.path.dirname(hpolib.benchmarks.ml.__file__)

        for _, pkg, _ in pkgutil.iter_modules([path, ]):
            pkg_name = "hpolib.benchmarks.ml.{:s}".format(pkg)
            importlib.import_module(pkg_name)
            mod_name = sys.modules[pkg_name]

            # Find Abstract Benchmark
            abstract_class = None
            for name, obj in inspect.getmembers(mod_name,
                                                inspect.isclass):
                if issubclass(obj, AbstractBenchmark) and inspect.isclass(obj):
                    abstract_class = obj

            # Check whether all classes accept rng
            if abstract_class is not None:
                for name, obj in inspect.getmembers(mod_name,
                                                inspect.isclass):
                    if issubclass(obj, abstract_class) and inspect.isclass(obj):
                        print(name)
                        rng = np.random.RandomState(1)
                        b = getattr(mod_name, name)(rng=rng)
                        self.assertEqual(b.rng, rng)

                        b = getattr(mod_name, name)(rng=2)
                        self.assertEqual(b.rng.get_state()[1], rng.get_state()[1])


