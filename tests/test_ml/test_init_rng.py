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
from hpolib.benchmarks.ml.autosklearn_benchmark import \
    MulticlassClassificationBenchmark


class TestInitRng(unittest.TestCase):

    def test_init_rng(self):
        path = os.path.dirname(hpolib.benchmarks.ml.__file__)

        for _, pkg, _ in pkgutil.iter_modules([path, ]):
            pkg_name = "hpolib.benchmarks.ml.{:s}".format(pkg)
            importlib.import_module(pkg_name)
            mod_name = sys.modules[pkg_name]

            # Find benchmarks that directly inherit from Abstract benchmark
            # to identify actual implementations
            abstract_class = None
            for name, obj in inspect.getmembers(mod_name,
                                                inspect.isclass):
                # Iterate over all classes in that module
                if issubclass(obj, AbstractBenchmark) and \
                        inspect.isclass(obj) and \
                        AbstractBenchmark in obj.__bases__:
                    # If this class directly inherits from AbstractBenchmark we
                    # found what we were looking for and continue
                    abstract_class = obj
                    break

            # Check whether all classes accept rng
            if abstract_class is not None:
                for name, obj in inspect.getmembers(mod_name, inspect.isclass):
                    # Again loop over all classes in that file
                    if issubclass(obj, abstract_class) and \
                            inspect.isclass(obj) and \
                            abstract_class in obj.__bases__:
                        # If this class directly inherits from the abstract
                        # class found above, we found a benchmark implementation
                        print(obj)
                        self.assertIn("rng",
                                      inspect.signature(
                                              getattr(mod_name,
                                                      name)).parameters)
