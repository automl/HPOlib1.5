'''
@author: Stefan Staeglich
'''

from hpolib.container.client.abstract_benchmark import AbstractBenchmarkClient


class CountingOnes(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        self.bName = "CountingOnes"
        self._setup(imgName="Forrester", **kwargs)
