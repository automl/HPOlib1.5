import logging

import unittest
import unittest.mock

from ConfigSpace import Configuration
import numpy as np
import pynisher

import hpolib
import hpolib.benchmarks.ml
import hpolib.benchmarks.synthetic_functions
import hpolib.benchmarks.surrogates.exploring_openml


class TestRandomConfig(unittest.TestCase):

    def setUp(self):
        self.logger = logging.getLogger("TestRandomConfig")
        self.logger.setLevel(logging.DEBUG)

    def test_random_config_exploring_openml(self):
        for b in [
            hpolib.benchmarks.surrogates.exploring_openml.GLMNET_4134(n_splits=2, n_iterations=2),
            hpolib.benchmarks.surrogates.exploring_openml.RPART_4134(n_splits=2, n_iterations=2),
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
                f_max = b.get_empirical_f_max()

                res = obj(c)
                if res is not None:
                    self.assertTrue(np.isfinite(res['function_value']))
                    self.assertTrue(np.isfinite(res['cost']))
                    self.assertGreaterEqual(res['function_value'], f_opt)
                    self.assertLessEqual(res['function_value'], f_max)
                else:
                    self.assertTrue(
                        obj.exit_status in (
                            pynisher.TimeoutException,
                            pynisher.MemorylimitException,
                        ),
                        msg=str(obj.exit_status)
                    )
