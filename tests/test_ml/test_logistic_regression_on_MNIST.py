import unittest
import unittest.mock

import numpy as np

import hpolib
import hpolib.benchmarks.ml.logistic_regression
from hpolib.benchmarks.ml import logistic_regression


class LogisticRegression(unittest.TestCase):

    def test_init_LogisticRegression10CVOnMnist(self):
        b = logistic_regression.LogisticRegression10CVOnMnist(rng=23)

        # Assert basic options
        self.assertEqual(b.folds, 10)
        self.assertEqual(b.valid, None)
        self.assertEqual(b.valid_targets, None)

    @unittest.mock.patch.object(hpolib.benchmarks.ml.logistic_regression.
                                LogisticRegression10CVOnMnist,
                                "_train_model")
    @unittest.mock.patch.object(hpolib.benchmarks.ml.logistic_regression.
                                LogisticRegression,
                                "get_data")
    def test_basic_LogisticRegression10CVOnMnist(self, data_mock, train_mock):
        ret_dict = [[0.5, 0.25, 0.1], [1, 2, 3]]
        train_mock.return_value = ret_dict
        data_mock.return_value = (np.random.randn(50000, 28*28),
                                  np.random.randint(low=1, high=10,
                                                    size=(50000, )),
                                  np.random.randn(10000, 28*28),
                                  np.random.randint(low=1, high=10,
                                                    size=(10000, )),
                                  np.random.randn(10000, 28*28),
                                  np.random.randint(low=1, high=10,
                                                    size=(10000, )))
        b = logistic_regression.LogisticRegression10CVOnMnist(rng=23)
        res = b.objective_function([-3, 0.5, 100, 0.5], fold=4, rng=1)
        self.assertEqual(res["cost"], 3)
        self.assertEqual(res["function_value"], 0.1)

    @unittest.mock.patch.object(hpolib.benchmarks.ml.logistic_regression.
                                LogisticRegression10CVOnMnist,
                                "_train_model")
    @unittest.mock.patch.object(hpolib.benchmarks.ml.logistic_regression.
                                LogisticRegression10CVOnMnist,
                                "objective_function_test")
    def test_CV_LogisticRegression10CVOnMnist(self, test_mock, train_mock):
        train_mock.return_value = ([False, False], [False, False])
        test_mock.return_value = True

        b = logistic_regression.LogisticRegression10CVOnMnist()
        cs = b.get_configuration_space()
        cfg = cs.sample_configuration()
        self.assertRaises(AssertionError, b.objective_function,
                          configuration=cfg, fold=-1)
        self.assertRaises(AssertionError, b.objective_function,
                          configuration=cfg, fold=11)

        self.assertEqual(b.objective_function(configuration=cfg,
                                              fold=0)["function_value"], False)
        self.assertEqual(b.objective_function(configuration=cfg,
                                              fold=9)["function_value"], False)
        self.assertEqual(b.objective_function(configuration=cfg,
                                              fold=10), True)

    def test_LogisticRegressionOnMnist(self):
        b = logistic_regression.LogisticRegressionOnMnist(rng=23)

        # Assert basic options
        self.assertEqual(b.valid, None)
        self.assertEqual(b.valid_targets, None)

    @unittest.mock.patch.object(hpolib.benchmarks.ml.logistic_regression.
                                LogisticRegressionOnMnist,
                                "_train_model")
    @unittest.mock.patch.object(hpolib.benchmarks.ml.logistic_regression.
                                LogisticRegressionOnMnist,
                                "get_data")
    def test_LogisticRegressionOnMnist(self, data_mock, train_mock):
        ret_dict = [[0.5, 0.25, 0.1], [1, 2, 3]]
        train_mock.return_value = ret_dict
        data_mock.return_value = (np.random.randn(50000, 28*28),
                                  np.random.randint(low=1, high=10,
                                                    size=(50000, )),
                                  np.random.randn(10000, 28*28),
                                  np.random.randint(low=1, high=10,
                                                    size=(10000, )),
                                  np.random.randn(10000, 28*28),
                                  np.random.randint(low=1, high=10,
                                                    size=(10000, )))
        b = logistic_regression.LogisticRegressionOnMnist(rng=23)
        res = b.objective_function([-3, 0.5, 100, 0.5], rng=1)
        self.assertEqual(res["cost"], 3)
        self.assertEqual(res["function_value"], 0.1)
