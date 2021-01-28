'''
@author: Stefan Staeglich
'''

from hpolib.container.client.abstract_benchmark import AbstractBenchmarkClient


class Branin(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        self.bName = "Branin"
        self._setup(**kwargs)
