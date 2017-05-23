import logging

from smac.tae.execute_func import ExecuteTAFuncDict
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC

from hpolib.benchmarks.ml import logistic_regression

import functools


def mini_wrapper(configuration, bench, **kwargs):
    """ mini wrapper to organize arguments """
    res_dict = bench.objective_function(configuration,
                                        fold=kwargs["instance"],
                                        rng=kwargs["seed"])
    return res_dict["function_value"]

logReg = logistic_regression.LogisticRegression10CVOnMnist(rng=10)

logger = logging.getLogger("Optimizer")  # Enable to show Debug outputs
logging.basicConfig(level=logging.DEBUG)

# Get Configuration Space
cs = logReg.get_configuration_space()

# Prepare TA
ta_call = functools.partial(mini_wrapper, bench=logReg)
taf = ExecuteTAFuncDict(ta_call)

# SMAC scenario object
scenario = Scenario({"run_obj": "quality",
                     "runcount-limit": 5,
                     "cs": cs,
                     "deterministic": "true",
                     "instances": list([str(i) for i in range(10)]), # train_inst_fh.name,
                     "test_instances": ["10", ]
                     })

# Evaluate default config as a test
def_value = taf.run(cs.get_default_configuration(), instance=0)[1]
print("Default Value: %.2f" % (def_value))

# Use SMAC to optimize function
smac = SMAC(scenario=scenario, rng=1,
            tae_runner=taf)
try:
    incumbent = smac.optimize()
finally:
    incumbent = smac.solver.incumbent

# Then evaluate final configuration
inc_value = taf.run(incumbent)[1]
print("Optimized Value: %.2f" % inc_value)
