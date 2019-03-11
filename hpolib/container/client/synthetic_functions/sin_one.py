'''
@author: Stefan Staeglich
'''

from hpolib.container.client.abstract_benchmark import AbstractBenchmarkClient


class SinOne(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        self.bName = "SinOne"
        self._setup(imgName="Forrester", **kwargs)

