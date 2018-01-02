import time
import numpy as np
import ConfigSpace as CS
import lasagne

from scipy import stats

from robo.models.bnn import BayesianNeuralNetwork
from sgmcmc.bnn.lasagne_layers import AppendLayer

from hpolib.abstract_benchmark import AbstractBenchmark
from hpolib.util import rng_helper
from functools import partial


def get_net(n_inputs, n_units_1, n_units_2):
    l_in = lasagne.layers.InputLayer(shape=(None, n_inputs))

    fc_layer_1 = lasagne.layers.DenseLayer(
        l_in,
        num_units=n_units_1,
        W=lasagne.init.HeNormal(),
        b=lasagne.init.Constant(val=0.0),
        nonlinearity=lasagne.nonlinearities.tanh)
    fc_layer_2 = lasagne.layers.DenseLayer(
        fc_layer_1,
        num_units=n_units_2,
        W=lasagne.init.HeNormal(),
        b=lasagne.init.Constant(val=0.0),
        nonlinearity=lasagne.nonlinearities.tanh)
    l_out = lasagne.layers.DenseLayer(
        fc_layer_2,
        num_units=1,
        W=lasagne.init.HeNormal(),
        b=lasagne.init.Constant(val=0.0),
        nonlinearity=lasagne.nonlinearities.linear)

    network = AppendLayer(l_out, num_units=1, b=lasagne.init.Constant(np.log(1e-3)))

    return network


class BNN(AbstractBenchmark):
    """

    """

    def __init__(self, rng=None):
        """
        Parameters
        ----------
        rng: int/None/RandomState
            set up rng
        """

        self.train, self.train_targets, self.valid, self.valid_targets, \
        self.test, self.test_targets = self.get_data()

        super(BNN, self).__init__()

        self.n_calls = 0
        self.rng = rng_helper.create_rng(rng)

    def get_data(self):
        raise NotImplementedError()

    @AbstractBenchmark._check_configuration
    def objective_function(self, config, step=None, **kwargs):
        st = time.time()

        net = partial(get_net, n_units_1=config['n_units_1'], n_units_2=config['n_units_2'])
        model = BayesianNeuralNetwork(sampling_method="sghmc",
                                      get_net=net,
                                      l_rate=config['l_rate'],
                                      mdecay=config['mdecay'],
                                      burn_in=config['burn_in'],
                                      n_iters=config['n_iters'] + config['burn_in'],
                                      precondition=True,
                                      normalize_input=True,
                                      normalize_output=True)
        model.train(self.train, self.train_targets)

        mean_pred, var_pred = model.predict(self.valid)

        # Negative log-likelihood
        y = - np.sum([stats.norm.pdf(self.valid_targets[i],
                                     loc=mean_pred[i],
                                     scale=np.sqrt(var_pred[i]))
                      for i in range(self.valid_targets.shape[0])])

        cost = time.time() - st
        return {'function_value': y, "cost": cost}

    @AbstractBenchmark._check_configuration
    def objective_function_test(self, config, **kwargs):
        st = time.time()

        net = partial(get_net, n_units_1=config['n_units_1'], n_units_2=config['n_units_2'])
        model = BayesianNeuralNetwork(sampling_method="sghmc",
                                      get_net=net,
                                      l_rate=config['l_rate'],
                                      mdecay=config['mdecay'],
                                      burn_in=config['burn_in'],
                                      n_iters=config['n_iters'],
                                      precondition=True,
                                      normalize_input=True,
                                      normalize_output=True)
        train = np.concatenate((self.train, self.valid))
        train_targets = np.concatenate((self.train_targets, self.valid_targets))
        model.train(train, train_targets)

        mean_pred, var_pred = model.predict(self.test)

        # Negative log-likelihood
        y = - np.sum([stats.norm.pdf(self.test_targets[i],
                                     loc=mean_pred[i],
                                     scale=np.sqrt(var_pred[i]))
                      for i in range(self.test_targets.shape[0])])

        cost = time.time() - st
        return {'function_value': y, "cost": cost}

    @staticmethod
    def get_configuration_space():
        cs = CS.ConfigurationSpace()

        cs.add_hyperparameter(CS.UniformFloatHyperparameter('l_rate',
                                                            lower=1e-6,
                                                            upper=1e-1,
                                                            default_value=1e-2,
                                                            log=True))

        cs.add_hyperparameter(CS.UniformIntegerHyperparameter('burn_in',
                                                              lower=500,
                                                              upper=10000,
                                                              default_value=3000))

        cs.add_hyperparameter(CS.UniformIntegerHyperparameter('n_iters',
                                                              lower=500,
                                                              upper=10000,
                                                              default_value=5000))

        cs.add_hyperparameter(CS.UniformIntegerHyperparameter('n_units_1',
                                                              lower=16,
                                                              upper=512,
                                                              default_value=64,
                                                              log=True))

        cs.add_hyperparameter(CS.UniformIntegerHyperparameter('n_units_2',
                                                              lower=16,
                                                              upper=512,
                                                              default_value=64,
                                                              log=True))

        cs.add_hyperparameter(CS.UniformFloatHyperparameter('mdecay',
                                                            lower=0,
                                                            upper=1,
                                                            default_value=0.05))

        return cs

    @staticmethod
    def get_meta_information():
        return {'name': 'BNN Benchmark',
                'references': []
                }


class BNNOnToyFunction(BNN):
    def get_data(self):
        rng = np.random.RandomState(42)

        def f(x):
            eps = rng.randn() * 0.02
            y = x + 0.3 * np.sin(2 * np.pi * (x + eps)) + 0.3 * np.sin(4 * np.pi * (x + eps)) + eps
            return y

        X = rng.rand(1000, 1)
        y = np.array([f(xi) for xi in X])[:, 0]

        train = X[:600]
        train_targets = y[:600]
        valid = X[600:800]
        valid_targets = y[600:800]
        test = X[800:]
        test_targets = y[800:]
        return train, train_targets, valid, valid_targets, test, test_targets


class BNNOnBostonHousing(BNN):
    def get_data(self):
        raise NotImplementedError
        # return train, train_targets, valid, valid_targets, test, test_targets
