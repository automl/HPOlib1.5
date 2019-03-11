import time
import numpy as np

try:
    from hpolib.benchmarks.ml.autosklearn_benchmark import fri_c1_1000_25
except ModuleNotFoundError:
    from hpolib.container.client.ml.autosklearn_benchmark import fri_c1_1000_25

import ConfigSpace


# Perform random search on the Branin function

b = fri_c1_1000_25()
print(b.get_meta_information())
start = time.time()

values = []

cs = b.get_configuration_space()

for i in range(2):
    configuration = cs.sample_configuration()
    # s = time.time()
    rval = b.objective_function(configuration, fold=1, folds=1)
    print(rval)
    print(type(rval['status']))
    # print("Done, took %.2f s" % ((time.time() - s)))
    loss = rval['function_value']
    values.append(loss)

print(np.min(values))

print("Done, took totally %.2f s" % ((time.time() - start)))

