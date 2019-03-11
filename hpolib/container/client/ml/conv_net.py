'''
@author: Stefan Staeglich
'''

import numpy
from hpolib.container.client.abstract_benchmark import AbstractBenchmarkClient


class ConvolutionalNeuralNetwork(AbstractBenchmarkClient):
    def _convertResultData(self, result):
        result['train_loss'] = numpy.array(result['train_loss'])
        result['valid_loss'] = numpy.array(result['valid_loss'])
        result['learning_curve'] = numpy.array(result['learning_curve'])
        result['learning_curve_cost'] = numpy.array(result['learning_curve_cost'])
        return result

    def objective_function(self, x, **kwargs):
        return self._convertResultData(super().objective_function(x, **kwargs))

    def objective_function_test(self, x, **kwargs):
        return self._convertResultData(super().objective_function_test(x, **kwargs))

    def test(self, *args, **kwargs):
        return self._convertResultData(super().objective_function_test(*args, **kwargs))


class ConvolutionalNeuralNetworkOnMNIST(ConvolutionalNeuralNetwork):
    def __init__(self, **kwargs):
        self.bName = "ConvolutionalNeuralNetworkOnMNIST"
        self._setup(gpu=True, **kwargs)


class ConvolutionalNeuralNetworkOnCIFAR10(ConvolutionalNeuralNetwork):
    def __init__(self, **kwargs):
        self.bName = "ConvolutionalNeuralNetworkOnCIFAR10"
        self._setup(gpu=True, **kwargs)


class ConvolutionalNeuralNetworkOnSVHN(ConvolutionalNeuralNetwork):
    def __init__(self, **kwargs):
        self.bName = "ConvolutionalNeuralNetworkOnSVHN"
        self._setup(gpu=True, **kwargs)
