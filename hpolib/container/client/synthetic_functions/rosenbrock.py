'''
@author: Stefan Staeglich
'''

from hpolib.container.client.abstract_benchmark import AbstractBenchmarkClient


class Rosenbrock2D(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        self.bName = "Rosenbrock2D"
        self._setup(imgName="Forrester", **kwargs)


class MultiFidelityRosenbrock2D(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        self.bName = "MultiFidelityRosenbrock2D"
        self._setup(imgName="Forrester", **kwargs)

