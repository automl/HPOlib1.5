#!/usr/bin/env python3

'''
@author: Stefan Staeglich
'''

import enum
import numpy
import os
import random
import string
import sys
import json

import Pyro4

import ConfigSpace as CS
from ConfigSpace.read_and_write import json as csjson

from hpolib.config import HPOlibConfig


class BenchmarkEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        if isinstance(obj, enum.Enum):
            return str(obj)
        return json.JSONEncoder.default(self, obj)


@Pyro4.expose
@Pyro4.behavior(instance_mode="single")
class BenchmarkServer():
    def __init__(self, socketId):
        self.pyroRunning = True
        config = HPOlibConfig()
        self.logger = config.logger

        self.socketId = socketId
        socketPath = config.socket_dir + self.socketId + "_unix.sock"
        if os.path.exists(socketPath):
            os.remove(socketPath)
        self.daemon = Pyro4.Daemon(unixsocket=socketPath)

        uri = self.daemon.register(self, self.socketId + ".unixsock")
        # start the event loop of the server to wait for calls
        self.daemon.requestLoop(loopCondition=lambda: self.pyroRunning)

    def initBenchmark(self, kwargsStr):
        if kwargsStr != "{}":
            kwargs = json.loads(kwargsStr)
            if 'rng' in kwargs and type(kwargs['rng']) == list:
                (rnd0, rnd1, rnd2, rnd3, rnd4) = kwargs['rng']
                rnd1 = [numpy.uint32(number) for number in rnd1]
                kwargs['rng'] = numpy.random.set_state((rnd0, rnd1, rnd2, rnd3, rnd4))
            self.b = Benchmark(**kwargs)
        else:
            self.b = Benchmark()

    def get_configuration_space(self):
        result = self.b.get_configuration_space()
        return csjson.write(result, indent=None)

    def objective_function_list(self, xString, kwargsStr):
        x = json.loads(xString)
        result = self.b.objective_function(x, **json.loads(kwargsStr))
        return json.dumps(result, indent=None, cls=BenchmarkEncoder)

    def objective_function(self, cString, csString, kwargsStr):
        cDict = json.loads(cString)
        cs = csjson.read(csString)
        configuration = CS.Configuration(cs, cDict)
        result = self.b.objective_function(configuration, **json.loads(kwargsStr))
        # Handle SMAC status
        return json.dumps(result, indent=None, cls=BenchmarkEncoder)

    def objective_function_test_list(self, xString, kwargsStr):
        x = json.loads(xString)
        result = self.b.objective_function_test(x, **json.loads(kwargsStr))
        # Handle SMAC runhistory
        return json.dumps(result, indent=None, cls=BenchmarkEncoder)

    def objective_function_test(self, cString, csString, kwargsStr):
        cDict = json.loads(cString)
        cs = csjson.read(csString)
        configuration = CS.Configuration(cs, cDict)
        result = self.b.objective_function_test(configuration, **json.loads(kwargsStr))
        # Handle SMAC runhistory
        return json.dumps(result, indent=None, cls=BenchmarkEncoder)

    def test(self, argsStr, kwargsStr):
        result = self.b.test(*json.loads(argsStr), **json.loads(kwargsStr))
        # Handle SMAC runhistory
        return json.dumps(result, indent=None, cls=BenchmarkEncoder)

    def get_meta_information(self):
        return json.dumps(self.b.get_meta_information(), indent=None)

    @Pyro4.oneway   # in case call returns much later than daemon.shutdown
    def shutdown(self):
        self.logger.debug('shutting down...')
        Pyro4.config.COMMTIMEOUT = 0.5
        self.pyroRunning = False
        self.daemon.shutdown()

if __name__ == "__main__":
    Pyro4.config.REQUIRE_EXPOSE = False

    if len(sys.argv) != 4:
        print("Usage: server.py <importBase> <benchmark> <socketId>")
        sys.exit()
    importBase = sys.argv[1]
    benchmark = sys.argv[2]
    socketId = sys.argv[3]

    exec("from hpolib.benchmarks.%s import %s as Benchmark" % (importBase, benchmark))
    bp = BenchmarkServer(socketId)
