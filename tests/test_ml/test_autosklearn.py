import unittest
import unittest.mock

import numpy as np
import openml

from autosklearn.data.xy_data_manager import XYDataManager
from autosklearn.constants import *
import hpolib.benchmarks.ml.autosklearn_benchmark


class TestAutoSklearnBenchmark(unittest.TestCase):

    def setUp(self):
        # Readonly API key for unit tests from Matthias Feurer
        openml.config.apikey = '953f6621518c13791dbbfc6d3698f5ad'

    @unittest.mock.patch.multiple(hpolib.benchmarks.ml.autosklearn_benchmark.AutoSklearnBenchmark, __abstractmethods__=set())
    @unittest.mock.patch('hpolib.benchmarks.ml.autosklearn_benchmark.AutoSklearnBenchmark._check_dependencies')
    @unittest.mock.patch('hpolib.benchmarks.ml.autosklearn_benchmark.AutoSklearnBenchmark._get_data_manager')
    @unittest.mock.patch('hpolib.benchmarks.ml.autosklearn_benchmark.AutoSklearnBenchmark._setup_backend')
    @unittest.mock.patch('hpolib.abstract_benchmark.AbstractBenchmark.__init__')
    def test_init(self, super_init_mock, setup_backend_mock,
                  get_data_manager_mock, check_dependencies_mock):
        fixture = 'sentinel'
        get_data_manager_mock.return_value = fixture
        auto = hpolib.benchmarks.ml.autosklearn_benchmark.AutoSklearnBenchmark(1)
        self.assertEqual(super_init_mock.call_count, 1)
        self.assertEqual(setup_backend_mock.call_count, 1)
        self.assertEqual(get_data_manager_mock.call_count, 1)
        self.assertEqual(check_dependencies_mock.call_count, 1)

    @unittest.mock.patch.multiple(hpolib.benchmarks.ml.autosklearn_benchmark.AutoSklearnBenchmark, __abstractmethods__=set())
    @unittest.mock.patch('hpolib.benchmarks.ml.autosklearn_benchmark.AutoSklearnBenchmark.__init__',
                         unittest.mock.Mock(return_value=None))
    def test_get_data_manager(self):
        # Test an allowed task - a task which is a 33% holdout task
        auto = hpolib.benchmarks.ml.autosklearn_benchmark.AutoSklearnBenchmark()
        auto._get_data_manager(289)
        self.assertIsInstance(auto.data_manager, XYDataManager)
        self.assertEqual(auto.data_manager.data['X_train'].shape, (101, 4))
        self.assertEqual(auto.data_manager.data['Y_train'].shape, (101,))
        self.assertEqual(auto.data_manager.data['X_test'].shape, (49, 4))
        self.assertEqual(auto.data_manager.data['Y_test'].shape, (49,))
        self.assertEqual(auto.data_manager.info['task'], MULTICLASS_CLASSIFICATION)
        self.assertEqual(auto.data_manager.feat_type, ['numerical', 'numerical',
                                                       'numerical', 'numerical'])

        # Test that tasks with more than one repeat fail
        auto = hpolib.benchmarks.ml.autosklearn_benchmark.AutoSklearnBenchmark()
        self.assertRaisesRegex(ValueError, 'Task 1939 has more than one repeat. '
                                           'This benchmark can only work with '
                                           'a single repeat.',
                               auto._get_data_manager, 1939)

        auto = hpolib.benchmarks.ml.autosklearn_benchmark.AutoSklearnBenchmark()
        self.assertRaisesRegex(ValueError, 'Task 10107 has more than one fold. '
                                           'This benchmark can only work with '
                                           'a single fold.',
                               auto._get_data_manager, 10107)




class TestIntegration(unittest.TestCase):

    def test_multiclass_on_iris(self):
        auto = hpolib.benchmarks.ml.autosklearn_benchmark.MulticlassClassificationBenchmark(289)
        all_rvals = []

        for i in range(10):
            print(i)
            train_rval, test_rval = auto.test(1, fold=i)
            for r in train_rval:
                print(r)
                all_rvals.append(r['function_value'])
            for r in test_rval:
                all_rvals.append(r['function_value'])

        self.assertLess(np.mean(all_rvals), 1.0)
        self.assertGreater(np.mean(all_rvals), 0.0)
        self.assertGreaterEqual(np.max(all_rvals), 0.0)
        self.assertLessEqual(np.max(all_rvals), 2.0)
        self.assertEqual(len(all_rvals), 20)
