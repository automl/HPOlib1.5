import hpolib.benchmarks.synthetic_functions as hpobench

try:
    from smac.facade.smac_facade import SMAC
    from smac.scenario.scenario import Scenario
except ImportError:
    print("To run this example you need to install SMAC")
    print("This can be done via `pip install SMAC`")

b = hpobench.Branin()
scenario = Scenario({"run_obj": "quality",
                     "runcount-limit": b.get_meta_information()['num_function_evals'],
                     "cs": b.get_configuration_space(),
                     "deterministic": "true"})
smac = SMAC(scenario=scenario, tae_runner=b)
x_star = smac.optimize()

print("Best value found:\n {:s}".format(str(x_star)))
print("with {:s}".format(str(b.objective_function(x_star))))
