import time
import numpy as np

# from hpolib.benchmarks.rl.carpole import CartpoleReduced
#from hpolib.benchmarks.bnn_benchmark import BNNOnYearPrediction
from hpolib.container.client.ml.bnn_benchmark import BNNOnYearPrediction

import ConfigSpace


# Perform random search on the Branin function

b = BNNOnYearPrediction()
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

