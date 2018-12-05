import unittest
import numpy as np

from hpolib.benchmarks import synthetic_functions
from hpolib.benchmarks.synthetic_functions.wrapper.transform_objective_value import Log10ObjectiveValue,\
    ExpObjectiveValue
from hpolib.benchmarks.synthetic_functions.wrapper.discretizer import DiscretizeDimensions


class TestAbstractBenchmark(unittest.TestCase):

    def test_log10_objective_value(self):
        f = synthetic_functions.Branin

        wrap_f = Log10ObjectiveValue(original_benchmark=f, rng=1)

        for x in f.get_meta_information()["optima"]:
            np.testing.assert_approx_equal(wrap_f.objective_function(x)["function_value"],
                                           np.log10(f.get_meta_information()["f_opt"]), significant=9)
            np.testing.assert_approx_equal(wrap_f.objective_function_test(x)["function_value"],
                                           np.log10(f.get_meta_information()["f_opt"]), significant=9)
        self.assertEqual(wrap_f.get_meta_information()["name"], "Log10ObjectiveValue(Branin)")

    def test_exp_objective_value(self):
        f = synthetic_functions.Branin

        wrap_f = ExpObjectiveValue(original_benchmark=f, rng=1)

        for x in f.get_meta_information()["optima"]:
            np.testing.assert_approx_equal(wrap_f.objective_function(x)["function_value"],
                                           np.exp(f.get_meta_information()["f_opt"]), significant=9)
            np.testing.assert_approx_equal(wrap_f.objective_function_test(x)["function_value"],
                                           np.exp(f.get_meta_information()["f_opt"]), significant=9)
        self.assertEqual(wrap_f.get_meta_information()["name"], "ExpObjectiveValue(Branin)")

    def test_disc_dimensions(self):
        f = synthetic_functions.Branin
        wrap_f = DiscretizeDimensions(original_benchmark=f, parameter=["x0"], steps=10, rng=1)

        cs = wrap_f.get_configuration_space()
        for c in cs.sample_configuration(3):
            wrap_f.objective_function(c)
        
        self.assertEqual(wrap_f.get_meta_information()["name"], "discrete(Branin)")


if __name__ == "__main__":
    unittest.main()

