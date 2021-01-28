import numpy as np
import random
import string
import sys
import time
from hpolib.benchmarks.rl.cartpole import CartpoleReduced
from hpolib.container.client.rl.cartpole import CartpoleReduced as CartpoleReducedContainer


#try:
from smac.facade.smac_facade import SMAC
from smac.scenario.scenario import Scenario
#except ImportError:
#    print("To run this example you need to install SMAC")
#    print("This can be done via `pip install SMAC`")


def main(b, rng, seed, dir, use_pynisher=True):
    # Runs SMAC on a given benchmark
    startTime = time.time()
    scenario = Scenario({
        "run_obj": "quality",
        "runcount-limit": 50,
        "cs": b.get_configuration_space(),
        "deterministic": "true",
        "output_dir": "./{:s}/run-{:d}/{:s}".format(b.get_meta_information()['name'],
                                               seed, dir)})


    smac = SMAC(scenario=scenario, tae_runner=b,
                rng=myrng)
    smac.solver.intensifier.tae_runner.use_pynisher = use_pynisher
    x_star = smac.optimize()
    endTime = time.time() - startTime
    print("Done, took totally %.2f s" % ((endTime)))

    print("Best value found:\n {:s}".format(str(x_star)))
    objFunc = b.objective_function(x_star)
    print("with {:s}".format(str(objFunc)))

if __name__ == "__main__":
    chars = string.ascii_uppercase + string.digits
    id = ''.join(random.choice(chars) for _ in range(10))

    seed = random.randint(1, 1001)
    print("Seed: %d" % seed)
    myrng = np.random.RandomState(seed)
    bc = CartpoleReducedContainer()
    bn = CartpoleReduced()
    print("Running as container:")
    main(bc, myrng, seed, id + "/container")
    #print("Running native without pynisher:")
    #main(bn, myrng, seed, id + "/pynisher", use_pynisher=False)
    print("Running native:")
    main(bn, myrng, seed, id + "/native")
