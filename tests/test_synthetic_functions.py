import unittest

from hpolib.benchmarks import synthetic_functions


class TestAbstractBenchmark(unittest.TestCase):

    def _test_synth_function(self, f):
        for x in f.get_meta_information()["optima"]:
            self.assertAlmostEqual(f(x), f.get_meta_information()["f_opt"], places=7)

        for x in f.get_meta_information()['bounds']:
            self.assertEqual(len(x), 2)

        self.assertEqual(len(f.get_meta_information()['bounds']),
                         len(f.get_configuration_space().get_hyperparameters()))

    def test_branin(self):
        f = synthetic_functions.Branin()
        self._test_synth_function(f)

    def test_hartmann3(self):
        f = synthetic_functions.Hartmann3()
        self._test_synth_function(f)

    def test_hartmann6(self):
        f = synthetic_functions.Hartmann6()
        self._test_synth_function(f)

    def test_camelback(self):
        f = synthetic_functions.Camelback()
        self._test_synth_function(f)

    def test_levy(self):
        for i in range(1, 11):
            f = getattr(synthetic_functions, "Levy%dD" % i)()
            self._test_synth_function(f)

    def test_goldstein_price(self):
        f = synthetic_functions.GoldsteinPrice()
        self._test_synth_function(f)

    def test_rosenbrock(self):
        for i in range(2, 5, 10):
            f = getattr(synthetic_functions, "Rosenbrock%dD" % i)()
            self._test_synth_function(f)

    def test_sin_one(self):
        f = synthetic_functions.SinOne()
        self._test_synth_function(f)

    def test_sin_two(self):
        f = synthetic_functions.SinTwo()
        self._test_synth_function(f)

if __name__ == "__main__":
    unittest.main()
