'''
@author: Stefan Staeglich
'''

from smac.tae.execute_ta_run import StatusType

from hpolib.container.client.abstract_benchmark import AbstractBenchmarkClient


class AutoSklearnBenchmark(AbstractBenchmarkClient):
    def objective_function(self, x, **kwargs):
        result = super().objective_function(x, **kwargs)
        result['status'] = eval(result['status'])
        return result

    def objective_function_test(self, x, **kwargs):
        result = super().objective_function_test(x, **kwargs)
        result['status'] = eval(result['status'])
        return result

    def test(self, *args, **kwargs):
        result = super().objective_function_test(*args, **kwargs)
        result['status'] = eval(result['status'])
        return result


class Sick(AutoSklearnBenchmark):
    def __init__(self, **kwargs):
        self.bName = "Sick"
        self._setup(imgName="AutoSklearnBenchmark", **kwargs)


class Splice(AutoSklearnBenchmark):
    def __init__(self, **kwargs):
        self.bName = "Splice"
        self._setup(imgName="AutoSklearnBenchmark", **kwargs)


class Adult(AutoSklearnBenchmark):
    def __init__(self, **kwargs):
        self.bName = "Adult"
        self._setup(imgName="AutoSklearnBenchmark", **kwargs)


class KROPT(AutoSklearnBenchmark):
    def __init__(self, **kwargs):
        self.bName = "KROPT"
        self._setup(imgName="AutoSklearnBenchmark", **kwargs)


class MNIST(AutoSklearnBenchmark):
    def __init__(self, **kwargs):
        self.bName = "MNIST"
        self._setup(imgName="AutoSklearnBenchmark", **kwargs)


class Quake(AutoSklearnBenchmark):
    def __init__(self, **kwargs):
        self.bName = "Quake"
        self._setup(imgName="AutoSklearnBenchmark", **kwargs)


class fri_c1_1000_25(AutoSklearnBenchmark):
    def __init__(self, **kwargs):
        self.bName = "fri_c1_1000_25"
        self._setup(imgName="AutoSklearnBenchmark", **kwargs)


class PC4(AutoSklearnBenchmark):
    def __init__(self, **kwargs):
        self.bName = "PC4"
        self._setup(imgName="AutoSklearnBenchmark", **kwargs)


class KDDCup09_appetency(AutoSklearnBenchmark):
    def __init__(self, **kwargs):
        self.bName = "KDDCup09_appetency"
        self._setup(imgName="AutoSklearnBenchmark", **kwargs)


class MagicTelescope(AutoSklearnBenchmark):
    def __init__(self, **kwargs):
        self.bName = "MagicTelescope"
        self._setup(imgName="AutoSklearnBenchmark", **kwargs)


class OVABreast(AutoSklearnBenchmark):
    def __init__(self, **kwargs):
        self.bName = "OVABreast"
        self._setup(imgName="AutoSklearnBenchmark", **kwargs)


class Covertype(AutoSklearnBenchmark):
    def __init__(self, **kwargs):
        self.bName = "Covertype"
        self._setup(imgName="AutoSklearnBenchmark", **kwargs)


class FBIS_WC(AutoSklearnBenchmark):
    def __init__(self, **kwargs):
        self.bName = "FBIS_WC"
        self._setup(imgName="AutoSklearnBenchmark", **kwargs)
