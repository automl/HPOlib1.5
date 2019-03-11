'''
@author: Stefan Staeglich
'''

from hpolib.container.client.abstract_benchmark import AbstractBenchmarkClient


class ClassificationNeuralNetwork(AbstractBenchmarkClient):
    def __init__(self, task_id, **kwargs):
        self.bName = "ClassificationNeuralNetwork"
        # Add task_id to kwargs
        kwargs["task_id"] = task_id
        self._setup(**kwargs)
