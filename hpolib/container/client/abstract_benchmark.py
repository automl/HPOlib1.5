'''
@author: Stefan Staeglich
'''

import abc
import json
import numpy
import os
import random
import signal
import string
import subprocess
import time

import Pyro4

from ConfigSpace.read_and_write import json as csjson

import hpolib.config


class AbstractBenchmarkClient(metaclass=abc.ABCMeta):
    def _setup(self, gpu=False, imgName=None, **kwargs):
        # Create unique ID
        self.socketId = self._id_generator()
        self.config = hpolib.config._config

        # Default image name is benchmark name
        if imgName is None:
            imgName = self.bName

        subprocess.run("SINGULARITY_PULLFOLDER=%s singularity pull --name %s.simg %s:%s" % (self.config.image_dir, imgName, self.config.image_source, imgName.lower()),
                       shell=True)
        iOptions = self.config.image_dir + imgName + ".simg"
        sOptions = self.bName + " " + self.socketId
        # Option for enabling GPU support
        gpuOpt = ""
        if gpu:
            gpuOpt = "--nv "
        # By default use named singularity instances. There exist a config option to disable this behaviour
        if self.config.singularity_use_instances:
            subprocess.run("singularity instance.start %s%s %s" % (gpuOpt, iOptions, self.socketId), shell=True)
            subprocess.Popen("singularity run %sinstance://%s %s" % (gpuOpt, self.socketId, sOptions), shell=True)
        else:
            self.sProcess = subprocess.Popen("singularity run %s%s %s" % (gpuOpt, iOptions, sOptions), shell=True)

        Pyro4.config.REQUIRE_EXPOSE = False
        # Generate Pyro 4 URI for connecting to client
        self.uri = "PYRO:" + self.socketId + ".unixsock@./u:" + self.config.socket_dir + self.socketId + "_unix.sock"
        self.b = Pyro4.Proxy(self.uri)

        # Handle rng and other optional benchmark arguments
        if 'rng' in kwargs and type(kwargs['rng']) == numpy.random.RandomState:
            (rnd0, rnd1, rnd2, rnd3, rnd4) = kwargs['rng'].get_state()
            rnd1 = [int(number) for number in rnd1]
            kwargs['rng'] = (rnd0, rnd1, rnd2, rnd3, rnd4)
        kwargsStr = json.dumps(kwargs)
        # Try to connect to server calling benchmark constructor via RPC. There exist a time limit
        self.config.logger.debug("Check connection to container and init benchmark")
        wait = 0
        while True:
            try:
                self.b.initBenchmark(kwargsStr)
            except Pyro4.errors.CommunicationError:
                self.config.logger.debug("Still waiting")
                time.sleep(5)
                wait += 5
                if wait < self.config.pyro_connect_max_wait:
                    continue
                else:
                    self.config.logger.debug("Waiting time exceeded. To high it up, adjust config option pyro_connect_max_wait.")
                    raise
            break
        self.config.logger.debug("Connected to container")

    def objective_function(self, x, **kwargs):
        # Create the arguments as Str
        if (type(x) is list):
            xString = json.dumps(x, indent=None)
            jsonStr = self.b.objective_function_list(xString, json.dumps(kwargs))
            return json.loads(jsonStr)
        else:
            # Create the arguments as Str
            cString = json.dumps(x.get_dictionary(), indent=None)
            csString = csjson.write(x.configuration_space, indent=None)
            jsonStr = self.b.objective_function(cString, csString, json.dumps(kwargs))
            return json.loads(jsonStr)

    def objective_function_test(self, x, **kwargs):
        # Create the arguments as Str
        if (type(x) is list):
            xString = json.dumps(x, indent=None)
            jsonStr = self.b.objective_function_test_list(xString, json.dumps(kwargs))
            return json.loads(jsonStr)
        else:
            # Create the arguments as Str
            cString = json.dumps(x.get_dictionary(), indent=None)
            csString = csjson.write(x.configuration_space, indent=None)
            jsonStr = self.b.objective_function_test(cString, csString, json.dumps(kwargs))
            return json.loads(jsonStr)

    def test(self, *args, **kwargs):
        result = self.b.test(json.dumps(args), json.dumps(kwargs))
        return json.loads(result)

    def get_configuration_space(self):
        jsonStr = self.b.get_configuration_space()
        return csjson.read(jsonStr)

    def get_meta_information(self):
        jsonStr = self.b.get_meta_information()
        return json.loads(jsonStr)

    def __call__(self, configuration, **kwargs):
        """ Provides interface to use, e.g., SciPy optimizers """
        return(self.objective_function(configuration, **kwargs)['function_value'])

    def _id_generator(self, size=6, chars=string.ascii_uppercase + string.digits):
        return ''.join(random.choice(chars) for _ in range(size))

    def __del__(self):
        Pyro4.config.COMMTIMEOUT = 1
        self.b.shutdown()
        if self.config.singularity_use_instances:
            subprocess.run("singularity instance.stop %s" % (self.socketId), shell=True)
        else:
            os.killpg(os.getpgid(self.sProcess.pid), signal.SIGTERM)
            self.sProcess.terminate()
        os.remove(self.config.socket_dir + self.socketId + "_unix.sock")
