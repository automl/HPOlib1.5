'''
@author: Stefan Staeglich
'''

from hpolib.container.client.abstract_benchmark import AbstractBenchmarkClient


class Forrester(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        self.bName = "Forrester"
        self._setup(**kwargs)
