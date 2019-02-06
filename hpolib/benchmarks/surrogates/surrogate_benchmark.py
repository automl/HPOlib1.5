import os
import pickle

from urllib.request import urlretrieve
import hpolib
from hpolib.abstract_benchmark import AbstractBenchmark


class SurrogateBenchmark(AbstractBenchmark):
    surrogate_base_url = 'https://ml.informatik.uni-freiburg.de/~sfalkner/surrogates/'

    def __init__(self, objective_surrogate_fn=None, cost_surrogate_fn=None, path=None, rng=None):

        super(SurrogateBenchmark, self).__init__(rng=rng)

        if path is None:
            self.surrogate_path = os.path.join(hpolib._config.data_dir, "Surrogates")
        else:
            self.surrogate_path = path

        self.surrogate_objective = None
        self.surrogate_cost = None

        if objective_surrogate_fn is not None:
            self.surrogate_objective = self.__load_surrogate(objective_surrogate_fn)

        if cost_surrogate_fn is not None:
            self.surrogate_cost = self.__load_surrogate(cost_surrogate_fn)

    def __load_surrogate(self, filename):
        os.makedirs(self.surrogate_path, exist_ok=True)
        
        fn = os.path.join(self.surrogate_path, filename)
        
        if not os.path.exists(fn):
            print("Downloading %s to %s" % (self.surrogate_base_url + filename, fn))
            
            urlretrieve(self.surrogate_base_url + filename, fn) 
        
        with open(fn, 'rb') as fh:
            surrogate = pickle.load(fh)

        return surrogate

    def objective_function(self, configuration, **kwargs):
        raise NotImplementedError()

    def objective_function_test(self, configuration, **kwargs):
        raise NotImplementedError()

