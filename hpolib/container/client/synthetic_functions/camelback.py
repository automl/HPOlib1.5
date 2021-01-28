'''
@author: Stefan Staeglich
'''

from hpolib.container.client.abstract_benchmark import AbstractBenchmarkClient


class Camelback(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        self.bName = "Camelback"
        self._setup(imgName="Forrester", **kwargs)

