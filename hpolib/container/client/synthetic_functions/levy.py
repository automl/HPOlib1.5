'''
@author: Stefan Staeglich
'''

from hpolib.container.client.abstract_benchmark import AbstractBenchmarkClient


class Levy1D(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        self.bName = "Levy1D"
        self._setup(imgName="Forrester", **kwargs)

