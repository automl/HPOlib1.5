import argparse
import os
import pickle
import sys
import time

import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.optimizers import BOHB as BOHB
from hpbandster.workers.hpolibbenchmark import HPOlib2Worker
from hpbandster.icml_2018_experiments.experiments.workers.cartpole import CartpoleReducedWorker as Worker

from hpolib.benchmarks.rl.cartpole import CartpoleReduced as Benchmark
from hpolib.container.client.rl.cartpole import CartpoleReduced as BenchmarkContainer

parser = argparse.ArgumentParser(description='Run BOHB on a HPOLib2 benchmark')
parser.add_argument('--min_budget',   type=int, help='Minimum budget used during the optimization.',    default=1)
parser.add_argument('--max_budget',   type=int, help='Maximum budget used during the optimization.',    default=9)
parser.add_argument('--n_iterations', type=int,   help='Number of iterations performed by the optimizer', default=16)
parser.add_argument('--n_workers', type=int,   help='Number of workers to run in parallel.', default=2)
parser.add_argument('--worker', help='Flag to turn this into a worker process', action='store_true')
parser.add_argument('--run_id', type=str, help='A unique run id for this optimization run. An easy option is to use the job id of the clusters scheduler.')
parser.add_argument('--nic_name',type=str, help='Which network interface to use for communication.')
parser.add_argument('--shared_directory',type=str, help='A directory that is accessible for all processes, e.g. a NFS share.')
parser.add_argument('--benchmark_type', type=str,   help='Benchmark type', default="container")

print("Parsing arguments")
args = parser.parse_args()

# Every process has to lookup the hostname
print("Lockup the hostname")
host = hpns.nic_name_to_host(args.nic_name)

# Benchmark
print("Init benchmark")
b = None
if args.benchmark_type == "container":
    b = BenchmarkContainer()
else:
    b = Benchmark()

if args.worker:
    print("Acting as worker")
    time.sleep(20)   # short artificial delay to make sure the nameserver is already running
    w = Worker(run_id=args.run_id, host=host)
    w.benchmark = b
    w.load_nameserver_credentials(working_directory=args.shared_directory)
    w.run(background=False)
    exit(0)

# Start a nameserver:
# We now start the nameserver with the host name from above and a random open port (by setting the port to 0)
print("Starting NameServer")
NS = hpns.NameServer(run_id=args.run_id, host=host, port=0, working_directory=args.shared_directory)
ns_host, ns_port = NS.start()

# Most optimizers are so computationally inexpensive that we can affort to run a
# worker in parallel to it. Note that this one has to run in the background to
# not plock!
print("Init worker")
w = Worker(run_id=args.run_id, host=host, nameserver=ns_host, nameserver_port=ns_port)
w.benchmark = b
w.run(background=True)

# Run an optimizer
# We now have to specify the host, and the nameserver information
print("Init optimizer")
result_logger = hpres.json_result_logger(directory=args.shared_directory + "/" + args.benchmark_type, overwrite=False)
bohb = BOHB(configspace=b.get_configuration_space(),
            run_id=args.run_id,
            host=host,
            nameserver=ns_host,
            nameserver_port=ns_port,
            result_logger=result_logger,
            min_budget=args.min_budget,
            max_budget=args.max_budget)
print("Running optimizer")
res = bohb.run(n_iterations=args.n_iterations, min_n_workers=args.n_workers)

# In a cluster environment, you usually want to store the results for later analysis.
# One option is to simply pickle the Result object
print("Saving results")
with open(os.path.join(args.shared_directory, args.benchmark_type + '-results.pkl'), 'wb') as fh:
    pickle.dump(res, fh)

# Step 4: Shutdown
# After the optimizer run, we must shutdown the master and the nameserver.
print("Stopping BOHB")
bohb.shutdown(shutdown_workers=True)
NS.shutdown()
