'''
@author: Stefan Staeglich
'''

from hpolib.container.client.abstract_benchmark import AbstractBenchmarkClient


class Hartmann3(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        self.bName = "Hartmann3"
        self._setup(imgName="Forrester", **kwargs)

