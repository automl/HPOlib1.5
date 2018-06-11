import unittest
import hpolib
import hpolib.config
import hpolib.util.data_manager

import numpy as np
import os

class TestSurrogateCNN(unittest.TestCase):
    def test_surrogate_cnn(self):
        # will test download_surrogate here.
        url = "http://www.ml4aad.org/wp-content/uploads/2017/12/lcnet_datasets.zip"
        surrogate = hpolib.util.data_manager.SurrogateData(surrogate_file="test.pkl",
                                                           url=url, folder="lcnet_datasets/convnet_cifar10/")
        self.surrogate_objective = surrogate.load_objective()
        self.surrogate_cost = surrogate.load_cost()
        self.assertEqual(1,1)