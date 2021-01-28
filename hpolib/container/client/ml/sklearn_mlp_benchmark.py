'''
@author: Stefan Staeglich
'''

from hpolib.container.client.abstract_benchmark import AbstractBenchmarkClient


class MLPOnHiggs(AbstractBenchmarkClient):
    def __init__(self, task_id, **kwargs):
        self.bName = "MLPOnHiggs"
        self._setup(**kwargs)


class MLPOnMnist(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        self.bName = "MLPOnMnist"
        self._setup(**kwargs)


class MLPOnVehicle(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        self.bName = "MLPOnVehicle"
        self._setup(**kwargs)
