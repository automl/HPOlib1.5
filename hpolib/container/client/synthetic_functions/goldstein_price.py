'''
@author: Stefan Staeglich
'''

from hpolib.container.client.abstract_benchmark import AbstractBenchmarkClient


class GoldsteinPrice(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        self.bName = "GoldsteinPrice"
        self._setup(imgName="Forrester", **kwargs)

