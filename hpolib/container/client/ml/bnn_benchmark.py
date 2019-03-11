'''
@author: Stefan Staeglich
'''

from hpolib.container.client.abstract_benchmark import AbstractBenchmarkClient


class BNNOnToyFunction(AbstractBenchmarkClient):
    def __init__(self, task_id, **kwargs):
        self.bName = "BNNOnToyFunction"
        self._setup(gpu=True, **kwargs)


class BNNOnBostonHousing(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        self.bName = "BNNOnBostonHousing"
        self._setup(gpu=True, **kwargs)


class BNNOnProteinStructure(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        self.bName = "BNNOnProteinStructure"
        self._setup(gpu=True, **kwargs)


class BNNOnYearPrediction(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        self.bName = "BNNOnYearPrediction"
        self._setup(gpu=True, **kwargs)
