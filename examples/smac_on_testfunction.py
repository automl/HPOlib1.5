import numpy as np
import sys
import hpolib.benchmarks.synthetic_functions as hpobench

try:
    from smac.facade.smac_facade import SMAC
    from smac.scenario.scenario import Scenario
except ImportError:
    print("To run this example you need to install SMAC")
    print("This can be done via `pip install SMAC`")


def main(b, seed):
    # Runs SMAC on a given benchmark
    scenario = Scenario({
        "run_obj": "quality",
        "runcount-limit": b.get_meta_information()['num_function_evals'],
        "cs": b.get_configuration_space(),
        "deterministic": "true",
        "output_dir": "./{:s}/run-{:d}".format(b.get_meta_information()['name'],
                                               seed)})

    smac = SMAC(scenario=scenario, tae_runner=b,
                rng=np.random.RandomState(seed))
    x_star = smac.optimize()

    print("Best value found:\n {:s}".format(str(x_star)))
    print("with {:s}".format(str(b.objective_function(x_star))))


if __name__ == "__main__":
    """
     This script tries to use the first given argument as a benchmark and the
     second as a seed, e.g.

     python smac_on_testfunction.py Branin 23
     python smac_on_testfunction.py hartmann3 23
    """

    try:
        benchmark = getattr(hpobench, sys.argv[1])()
    except AttributeError:
        print("Use Branin")
        benchmark = hpobench.Branin()

    try:
        seed = int(float(sys.argv[2]))
    except:
        seed = np.random.randn(1, 1)

    main(b=benchmark, seed=seed)
