'''
@author: Stefan Staeglich
'''

from hpolib.container.client.abstract_benchmark import AbstractBenchmarkClient


class SinTwo(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        self.bName = "SinTwo"
        self._setup(imgName="Forrester", **kwargs)

