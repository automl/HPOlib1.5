import keras
import numpy as np

from hpolib.util.data_manager import MNISTData
from hpolib.abstract_iterative_benchmark import AbstractIterativeBenchmark
from hpolib.iterative_benchmarks.iterative_fc_net import IterativeFCNet


class IterativeFCNetOnMNIST(AbstractIterativeBenchmark):

    def __init__(self):
        self.train, self.train_targets, self.valid, self.valid_targets, \
            self.test, self.test_targets = self.get_data()

        num_classes = np.int32(np.unique(self.train_targets).shape[0])

        if len(self.train_targets.shape) == 1:
            self.train_targets = keras.utils.to_categorical(self.train_targets, num_classes)
            self.valid_targets = keras.utils.to_categorical(self.valid_targets, num_classes)
            self.test_targets = keras.utils.to_categorical(self.test_targets, num_classes)

    def get_data(self):
        dm = MNISTData()
        return dm.load()

    def get_model(self, config=None):

        if config is None:
            config = IterativeFCNet.get_configuration_space().sample_configuration()

        return IterativeFCNet(config=config,
                              train=self.train,
                              train_targets=self.train_targets,
                              valid=self.valid,
                              valid_targets=self.valid_targets,
                              test=self.test,
                              test_targets=self.test_targets)

    @staticmethod
    def get_configuration_space():
        return IterativeFCNet.get_configuration_space()
