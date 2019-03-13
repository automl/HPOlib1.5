import importlib
import inspect
import logging
import pkgutil
import os
import sys

import unittest.mock

from ConfigSpace import Configuration
import numpy as np
import pynisher
import theano

import hpolib
import hpolib.benchmarks.ml
import hpolib.benchmarks.synthetic_functions
import hpolib.benchmarks.surrogates.exploring_openml
from hpolib.abstract_benchmark import AbstractBenchmark
from hpolib.benchmarks.ml.autosklearn_benchmark import AutoSklearnBenchmark
from hpolib.benchmarks.ml.autosklearn_benchmark import \
    MulticlassClassificationBenchmark
from hpolib.benchmarks.ml.logistic_regression import LogisticRegression
from hpolib.benchmarks.ml.fully_connected_network import FullyConnectedNetwork
from hpolib.benchmarks.ml.conv_net import ConvolutionalNeuralNetwork


class TestRandomConfig(unittest.TestCase):

    def setUp(self):
        self.logger = logging.getLogger("TestRandomConfig")
        self.logger.setLevel(logging.DEBUG)

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
                        print(obj, name)

                        if issubclass(obj, AutoSklearnBenchmark) and not \
                                MulticlassClassificationBenchmark in obj.__bases__:
                            # Special case for auto-sklearn which has
                            # two baseclasses
                            continue

                        if issubclass(obj, LogisticRegression):
                            # Special case for log reg as it does require
                            # different theano flags
                            theano.config.floatX = "float64"

                        if issubclass(obj, FullyConnectedNetwork) or \
                                issubclass(obj, ConvolutionalNeuralNetwork):
                            # Special case for networks as they require
                            # different theano flags to run on CPU
                            theano.config.floatX = "float32"
                            if sys.version_info > (3, 5, 0):
                                theano.config.optimizer = 'None'

                        b = getattr(mod_name, name)()
                        cfg = b.get_configuration_space()
                        for i in range(5):
                            c = cfg.sample_configuration()

                            # Limit Wallclocktime using pynisher
                            obj = pynisher.enforce_limits(
                                wall_time_in_s=10,
                                mem_in_mb=3000,
                                grace_period_in_s=5,
                                logger=self.logger
                            )(b.objective_function)
                            res = obj(c)
                            if res is not None:
                                self.assertTrue(np.isfinite(res['cost']))
                                self.assertTrue(np.isfinite(res['function_value']))
                            else:
                                self.assertTrue(
                                    obj.exit_status in (
                                        pynisher.TimeoutException,
                                        pynisher.MemorylimitException,
                                    ),
                                    msg=str(obj.exit_status)
                                )
            else:
                raise ValueError("{:s} does not contain a basic benchmark that is"
                                 " derived from AbstractBenchmark".
                                 format(mod_name))

    def test_random_config_exploring_openml(self):
        for b in [
            hpolib.benchmarks.surrogates.exploring_openml.GLMNET_4134(n_splits=2, n_iterations=2),
            hpolib.benchmarks.surrogates.exploring_openml.RPART_4134(n_splits=2, n_iterations=2),
            hpolib.benchmarks.surrogates.exploring_openml.KKNN_3(n_splits=2, n_iterations=2),
            hpolib.benchmarks.surrogates.exploring_openml.SVM_1486(n_splits=2, n_iterations=2),
            hpolib.benchmarks.surrogates.exploring_openml.Ranger_1043(n_splits=2, n_iterations=2),
            hpolib.benchmarks.surrogates.exploring_openml.XGBoost_4534(n_splits=2, n_iterations=2),
        ]:

            cfg = b.get_configuration_space()
            for i in range(5):
                c = cfg.sample_configuration()

                # Limit Wallclocktime using pynisher
                obj = pynisher.enforce_limits(
                    wall_time_in_s=10,
                    mem_in_mb=3000,
                    grace_period_in_s=5,
                    logger=self.logger
                )(b.objective_function)

                f_opt = b.get_empirical_f_opt()
                y_opt = b.objective_function(f_opt)['function_value']
                self.assertIsInstance(f_opt, Configuration)
                self.assertGreaterEqual(y_opt, 0)
                self.assertLessEqual(y_opt, 1)

                f_max = b.get_empirical_f_max()
                y_max = b.objective_function(f_max)['function_value']
                self.assertIsInstance(f_max, Configuration)
                self.assertGreaterEqual(y_max, 0)
                self.assertLessEqual(y_max, 1)

                res = obj(c)
                if res is not None:
                    self.assertTrue(np.isfinite(res['function_value']))
                    self.assertTrue(np.isfinite(res['cost']))
                    self.assertGreaterEqual(res['function_value'], y_opt)
                    self.assertLessEqual(res['function_value'], y_max)
                else:
                    self.assertTrue(
                        obj.exit_status in (
                            pynisher.TimeoutException,
                            pynisher.MemorylimitException,
                        ),
                        msg=str(obj.exit_status)
                    )
