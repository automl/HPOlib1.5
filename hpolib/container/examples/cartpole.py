import time
import numpy as np

# from hpolib.benchmarks.rl.cartpole import CartpoleReduced
from hpolib.container.client.rl.cartpole import CartpoleReduced

import ConfigSpace


# Perform random search on the Branin function

b = CartpoleReduced()
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

