'''
@author: Stefan Staeglich
'''

from hpolib.container.client.abstract_benchmark import AbstractBenchmarkClient


class Bohachevsky(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        self.bName = "Bohachevsky"
        self._setup(imgName="Forrester", **kwargs)
