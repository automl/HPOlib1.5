import time
import numpy as np
import lasagne
import theano
import theano.tensor as T

from sklearn.model_selection import StratifiedKFold

import ConfigSpace as CS

from hpolib.abstract_benchmark import AbstractBenchmark
from hpolib.util.data_manager import DataManager, \
    MNISTData, MNISTDataCrossvalidation
from hpolib.util import rng_helper


class LogisticRegression(AbstractBenchmark):
    """
        Logistic regression benchmark which resembles the benchmark used in
        the MTBO / Freeze Thaw paper. The hyperparameters are the learning
        rate (on log scale), L2 regularization, batch size and the dropout
        ratio on the inputs. The weights are optimized with stochastic
        gradient descent and we do NOT perform early stopping on
        the validation data set.


    """

    def __init__(self, rng=None):
        super(LogisticRegression, self).__init__(rng=rng)

        self.train, self.train_targets, self.valid, self.valid_targets, \
            self.test, self.test_targets = self.get_data()
        self.num_epochs = 100

        # Use 10 time the number of classes as lower bound for the dataset fraction
        self.num_classes = np.unique(self.train_targets).shape[0]
        self.s_min = 2000  # Minimum batch size

        lasagne.random.set_rng(self.rng)

    def get_data(self):
        raise NotImplementedError("Do not use this benchmark as this is only "
                                  "a skeleton for further implementations.")

    @AbstractBenchmark._check_configuration
    @AbstractBenchmark._configuration_as_array
    def objective_function(self, x, dataset_fraction=1, **kwargs):

        # Shuffle training data
        rng = kwargs.get("rng")
        self.rng = rng_helper.get_rng(rng=rng, self_rng=self.rng)
        # if rng was not not, set rng for lasagne
        if rng is not None:
            lasagne.random.set_rng(self.rng)

        # Shuffle training data
        shuffle = self.rng.permutation(self.train.shape[0])
        size = int(dataset_fraction * self.train.shape[0])

        # Split of dataset subset
        train = self.train[shuffle[:size]]
        train_targets = self.train_targets[shuffle[:size]]

        lc_curve, cost_curve = \
            self._train_model(config=x,
                              train=train,
                              train_targets=train_targets,
                              valid=self.valid,
                              valid_targets=self.valid_targets)
        y = lc_curve[-1]
        c = cost_curve[-1]

        return {'function_value': y, "cost": c,
                "learning_curve": lc_curve, "cost_curve": cost_curve}

    @AbstractBenchmark._check_configuration
    @AbstractBenchmark._configuration_as_array
    def objective_function_test(self, x, **kwargs):

        rng = kwargs.get("rng")
        self.rng = rng_helper.get_rng(rng=rng, self_rng=self.rng)
        # if rng was not None, set rng for lasagne
        if rng is not None:
            lasagne.random.set_rng(self.rng)

        train = np.concatenate((self.train, self.valid))
        train_targets = np.concatenate((self.train_targets, self.valid_targets))
        lc_curve, cost_curve = \
            self._train_model(config=x, train=train,
                              train_targets=train_targets,
                              valid=self.test,
                              valid_targets=self.test_targets)
        y = lc_curve[-1]
        c = cost_curve[-1]

        return {'function_value': y, "cost": c,
                "learning_curve": lc_curve, "cost_curve": cost_curve}

    @staticmethod
    def get_configuration_space():
        cs = CS.ConfigurationSpace(seed=np.random.randint(1, 100000))
        cs.generate_all_continuous_from_bounds(LogisticRegression.get_meta_information()['bounds'])
        return cs

    @staticmethod
    def get_meta_information():
        return {'name': 'Logistic Regression',
                'bounds': [[-6, 0],  # learning rate
                           [0, 1],  # l2 regularization
                           [20, 2000],  # batch size
                           [0, .75]],  # dropout rate
                'references': ["@article{klein-bnn16a,"
                               "author = {A. Klein and S. Falkner and T. Springenberg and F. Hutter},"
                               "title = {Bayesian Neural Network for Predicting Learning Curves},"
                               "booktitle = {NIPS 2016 Bayesian Neural Network Workshop},"
                               "month = dec,"
                               "year = {2016}}"]
                }

    def _train_model(self, config, train, train_targets, valid, valid_targets):
        """ helper method that accepts data and configuration, transforms
         hyperparmaters and returns lc_curve, cost_curve on valid data

        Parameters
        ----------
        config: list, np.ndarray, Configspace.config
            [log10(learning_rate), l2_reg, batch_size, dropout_rate]
        train: np.ndarray [n_samples, n_features]
        train_targets: np.ndarray [n_samples, n_objectives]
        valid: np.ndarray [n_samples, n_features]
        valid_targets: [n_samples, n_objectives]

        Returns
        -------
        learning_curve, cost
        """
        learning_rate = np.float32(10 ** config[0])
        l2_reg = np.float32(config[1])
        batch_size = np.int32(config[2])
        dropout_rate = np.float32(config[3])

        return self.run(train=train,
                        train_targets=train_targets,
                        valid=valid,
                        valid_targets=valid_targets,
                        learning_rate=learning_rate,
                        l2_reg=l2_reg,
                        batch_size=batch_size,
                        dropout_rate=dropout_rate,
                        num_epochs=self.num_epochs)

    def iterate_minibatches(self, inputs, targets, batch_size, shuffle=False):
        assert len(inputs) == len(targets)
        if shuffle:
            indices = np.arange(len(inputs))
            self.rng.shuffle(indices)
        for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)
            yield inputs[excerpt], targets[excerpt]

    def run(self, train, train_targets,
            valid, valid_targets,
            learning_rate=1.0, l2_reg=0.0,
            batch_size=200, dropout_rate=0.1, num_epochs=100):
        """ accepts data and hyperparameters and returns lc_curve, cost_curve

        Parameters
        ----------
        train: np.ndarray [n_samples, n_features]
        train_targets: np.ndarray [n_samples, n_objectives]
        valid: np.ndarray [n_samples, n_features]
        valid_targets: [n_samples, n_objectives]

        learning_rate: float
        l2_reg: float
        batch_size: float
        dropout_rate: float
        num_epochs: int

        Returns
        -------
        learning_curve, cost
        """
        start_time = time.time()

        input_var = T.dmatrix('inputs')
        target_var = T.ivector('targets')

        # Build net
        lr = lasagne.layers.InputLayer(shape=(None, train.shape[1]),
                                       input_var=input_var)

        lr = lasagne.layers.DropoutLayer(lr, p=dropout_rate)

        lr = lasagne.layers.DenseLayer(lr,
                                       num_units=self.num_classes,
                                       W=lasagne.init.HeNormal(),
                                       b=lasagne.init.Constant(val=0.0),
                                       nonlinearity=lasagne.nonlinearities.softmax)

        # Define Theano functions
        params = lasagne.layers.get_all_params(lr, trainable=True)
        prediction = lasagne.layers.get_output(lr)
        loss = lasagne.objectives.categorical_crossentropy(prediction,
                                                           target_var)
        # Add L2 regularization for the weights
        l2_penalty = l2_reg * lasagne.regularization.\
            regularize_network_params(lr, lasagne.regularization.l2)

        loss += l2_penalty
        loss = loss.mean()

        test_prediction = lasagne.layers.get_output(lr, deterministic=True)
        test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                                target_var)
        test_loss = test_loss.mean()

        test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                          dtype=theano.config.floatX)

        learning_rate = theano.shared(learning_rate)

        updates = lasagne.updates.sgd(loss, params, learning_rate=learning_rate)

        train_fn = theano.function([input_var, target_var], loss, updates=updates)

        val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

        print("Starting training...")

        learning_curve = np.zeros([num_epochs])
        cost = np.zeros([num_epochs])

        for e in range(num_epochs):

            train_err = 0
            train_batches = 0

            for batch in self.iterate_minibatches(train, train_targets,
                                                  batch_size, shuffle=True):
                inputs, targets = batch
                train_err += train_fn(inputs, targets)
                train_batches += 1

            val_err = 0
            val_acc = 0
            val_batches = 0
            for batch in self.iterate_minibatches(valid, valid_targets,
                                                  batch_size, shuffle=False):
                inputs, targets = batch
                err, acc = val_fn(inputs, targets)
                val_err += err
                val_acc += acc
                val_batches += 1

            learning_curve[e] = 1 - val_acc / val_batches
            cost[e] = time.time() - start_time

        return learning_curve, cost


class LogisticRegressionOnMnist(LogisticRegression):

    def get_data(self):
        dm = MNISTData()
        return dm.load()

    @staticmethod
    def get_meta_information():
        d = LogisticRegression.get_meta_information()
        d["references"].append("@article{lecun-ieee98,"
                               "title={Gradient-based learning applied to document recognition},"
                               "author={Y. LeCun and L. Bottou and Y. Bengio and P. Haffner},"
                               "journal={Proceedings of the IEEE},"
                               "pages={2278--2324},"
                               "year={1998},"
                               "publisher={IEEE}"
                               )
        return d


class LogisticRegression10CVOnMnist(LogisticRegressionOnMnist):

    def __init__(self, rng=None):
        super(LogisticRegression10CVOnMnist, self).__init__(rng=rng)
        self.folds = 10

    def get_data(self):
        dm = MNISTDataCrossvalidation()
        X_train, y_train, X_test, y_test = dm.load()
        return X_train, y_train, None, None, X_test, y_test

    @AbstractBenchmark._check_configuration
    @AbstractBenchmark._configuration_as_array
    def objective_function(self, x, **kwargs):
        """
        Evaluates dataset_fraction of one fold of a 10 fold CV

        :param x: array/Configurations
            configuration
        :param kwargs:
            fold: int in [0, 9]
                if fold == 10, return test performance
            rng: rng, int or None
                if not None overwrites current RandomState

        :return: dict
        """
        fold = int(float(kwargs["fold"]))
        assert 0 <= fold <= self.folds

        arg_rng = kwargs.get("rng", None)
        self.rng = rng_helper.get_rng(rng=arg_rng, self_rng=self.rng)

        # if arg_rng was not not, set rng for lasagne
        if arg_rng is not None:
            lasagne.random.set_rng(self.rng)

        if fold == self.folds:
            return self.objective_function_test(x, **kwargs)

        # Compute crossvalidation splits
        kf = StratifiedKFold(n_splits=self.folds, shuffle=True,
                             random_state=self.rng)

        # Get indices for required fold
        train_idx = None
        valid_idx = None
        for idx, split in enumerate(kf.split(X=self.train,
                                             y=self.train_targets)):
            if idx == fold:
                train_idx = split[0]
                valid_idx = split[1]
                break

        valid = self.train[valid_idx, :]
        valid_targets = self.train_targets[valid_idx]

        train = self.train[train_idx, :]
        train_targets = self.train_targets[train_idx]

        # Get performance
        lc_curve, cost_curve = self._train_model(config=x,
                                                 train=train,
                                                 train_targets=train_targets,
                                                 valid=valid,
                                                 valid_targets=valid_targets)
        y = lc_curve[-1]
        c = cost_curve[-1]
        return {'function_value': y, "cost": c, "learning_curve": lc_curve,
                "cost_curve": cost_curve}

    @AbstractBenchmark._check_configuration
    @AbstractBenchmark._configuration_as_array
    def objective_function_test(self, x, **kwargs):
        rng = kwargs.get("rng", None)
        self.rng = rng_helper.get_rng(rng=rng, self_rng=self.rng)

        if rng is not None:
            lasagne.random.set_rng(self.rng)

        lc_curve, cost_curve = \
            self._train_model(config=x, train=self.train,
                              train_targets=self.train_targets,
                              valid=self.test,
                              valid_targets=self.test_targets)
        y = lc_curve[-1]
        c = cost_curve[-1]

        return {'function_value': y, "cost": c,
                "learning_curve": lc_curve, "cost_curve": cost_curve}

    @staticmethod
    def get_meta_information():
        d = LogisticRegression.get_meta_information()
        d["references"].append("""@article{lecun-ieee98,
title={Gradient-based learning applied to document recognition},
author={Y. LeCun and L. Bottou and Y. Bengio and P. Haffner},
journal={Proceedings of the IEEE},
pages={2278--2324},
year={1998},
publisher={IEEE}
""")
        d["cvfolds"] = 10
        d["wallclocklimit"] = 24*60*60
        d['num_function_evals'] = np.inf
        d['cutoff'] = 1800
        return d
