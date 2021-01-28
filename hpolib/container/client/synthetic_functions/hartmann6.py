'''
@author: Stefan Staeglich
'''

from hpolib.container.client.abstract_benchmark import AbstractBenchmarkClient


class Hartmann6(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        self.bName = "Hartmann6"
        self._setup(imgName="Forrester", **kwargs)

