'''
@author: Stefan Staeglich
'''

from hpolib.container.client.abstract_benchmark import AbstractBenchmarkClient


class SvmOnMnist(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        self.bName = "SvmOnMnist"
        self._setup(**kwargs)


class SvmOnVehicle(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        self.bName = "SvmOnVehicle"
        self._setup(**kwargs)


class SvmOnCovertype(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        self.bName = "SvmOnCovertype"
        self._setup(**kwargs)


class SvmOnLetter(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        self.bName = "SvmOnLetter"
        self._setup(**kwargs)


class SvmOnAdult(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        self.bName = "SvmOnAdult"
        self._setup(**kwargs)


class SvmOnHiggs(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        self.bName = "SvmOnHiggs"
        self._setup(**kwargs)
