import numpy as np
import random
import string
import sys
import time
from hpolib.benchmarks.ml.conv_net import ConvolutionalNeuralNetworkOnMNIST
from hpolib.container.client.ml.conv_net import ConvolutionalNeuralNetworkOnMNIST as ConvolutionalNeuralNetworkOnMNISTContainer


try:
    from smac.facade.smac_facade import SMAC
    from smac.scenario.scenario import Scenario
except ImportError:
    print("To run this example you need to install SMAC")
    print("This can be done via `pip install SMAC`")


def main(b, rng, seed, dir):
    # Runs SMAC on a given benchmark
    startTime = time.time()
    scenario = Scenario({
        "run_obj": "quality",
        "runcount-limit": 20,
        "cs": b.get_configuration_space(),
        "deterministic": "true",
        "output_dir": "./{:s}/run-{:d}/{:s}".format(b.get_meta_information()['name'],
                                               seed, dir)})


    smac = SMAC(scenario=scenario, tae_runner=b,
                rng=myrng)
    x_star = smac.optimize()
    endTime = time.time() - startTime
    print("Done, took totally %.2f s" % ((endTime)))

    print("Best value found:\n {:s}".format(str(x_star)))
    objFunc = b.objective_function(x_star)
    print("with {:s}".format(str(objFunc)))

if __name__ == "__main__":
    chars = string.ascii_uppercase + string.digits
    id = ''.join(random.choice(chars) for _ in range(10))
    
    seed = random.randint(1, 101)
    print("Seed: %d" % seed)
    myrng = np.random.RandomState(seed)
    bc = ConvolutionalNeuralNetworkOnMNISTContainer()
    bn = ConvolutionalNeuralNetworkOnMNIST()
    print("Running as container:")
    #main(bc, myrng, seed, id + "/container")
    print("Running native:")
    main(bn, myrng, seed, id + "/native")
