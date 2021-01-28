import time
import numpy as np

# from hpolib.benchmarks.rl.carpole import CartpoleReduced
from hpolib.benchmarks.ml.conv_net import ConvolutionalNeuralNetworkOnMNIST
#from hpolib.container.client.ml.conv_net import ConvolutionalNeuralNetworkOnMNIST

import ConfigSpace


# Perform random search on the Branin function

b = ConvolutionalNeuralNetworkOnMNIST()
start = time.time()

values = []

cs = b.get_configuration_space()

for i in range(2):
    configuration = cs.sample_configuration()
    # s = time.time()
    rval = b.objective_function(configuration)
    # print("Done, took %.2f s" % ((time.time() - s)))
    loss = rval['function_value']
    values.append(loss)

print(np.min(values))

print("Done, took totally %.2f s" % ((time.time() - start)))

