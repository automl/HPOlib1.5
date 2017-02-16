from hpolib.benchmarks.synthetic_functions import Branin

from smac.facade.smac_facade import SMAC
from smac.scenario.scenario import Scenario

n_iters = 200

b= Branin()
scenario = Scenario({"run_obj": "quality",
                     "runcount-limit": n_iters,
                     "cs": b.get_configuration_space(),
                     "deterministic": "true"})

smac = SMAC(scenario=scenario, tae_runner=b)
x_star = smac.optimize()
