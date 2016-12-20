import os
import sys
import time

import numpy as np
import lasagne
import theano
import theano.tensor as T

import ConfigSpace as CS

from hpolib.abstract_benchmark import AbstractBenchmark
import hpolib


class FullyConnectedNetwork(AbstractBenchmark):

    def __init__(self, max_num_epochs=100, rng=None):
        """
        Initializes Fully connected network
        Parameters
        ----------

        max_num_epochs: int
            set maximum number of epochs. Needed to calculate how many number
            of epochs to use for training given number of steps in [0, 100].

        rng: str
            set up rng
        """
        self.path = hpolib._config.data_dir
        print("Store data in %s" % self.path)

        self.train, self.train_targets, self.valid, self.valid_targets, \
            self.test, self.test_targets = self.get_data()
        self.max_num_epochs = max_num_epochs

        self.num_classes = np.int32(np.unique(self.train_targets).shape[0])
        # Minimum dataset size is equal to the maximum batch size
        self.s_min = 512
        if rng is None:
            self.rng = np.random.RandomState()
        else:
            self.rng = rng

        lasagne.random.set_rng(self.rng)
        super(FullyConnectedNetwork, self).__init__()

    def get_data(self):
        pass

    @AbstractBenchmark._check_configuration
    @AbstractBenchmark._configuration_as_array
    def objective_function(self, x, dataset_fraction=1, steps=1, **kwargs):

        x = np.array(x, dtype=np.float32)

        num_epochs = int(1 + (self.max_num_epochs - 1) * steps)

        # Shuffle training data
        shuffle = self.rng.permutation(self.train.shape[0])
        size = int(dataset_fraction * self.train.shape[0])

        # Split of dataset subset
        train = self.train[shuffle[:size]]
        train_targets = self.train_targets[shuffle[:size]]

        lc_curve, cost_curve, train_loss, valid_loss = \
            self.train_net(train, train_targets,
                           self.valid, self.valid_targets,
                           init_learning_rate=np.float32(np.power(10., x[0])),
                           l2_reg=np.float32(np.power(10., x[1])),
                           batch_size=np.int32(x[2]),
                           gamma=np.float32(np.power(10, x[3])),
                           power=np.float32(x[4]),
                           momentum=np.float32(x[5]),
                           n_units_1=np.int32(np.power(2, x[6])),
                           n_units_2=np.int32(np.power(2, x[7])),
                           dropout_rate_1=np.float32(x[8]),
                           dropout_rate_2=np.float32(x[9]),
                           num_epochs=np.int32(num_epochs))

        y = lc_curve[-1]
        c = cost_curve[-1]
        return {'function_value': y,
                "cost": c,
                "learning_curve_valid_error": lc_curve,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "learning_curve_cost": cost_curve}

    @AbstractBenchmark._check_configuration
    @AbstractBenchmark._configuration_as_array
    def objective_function_test(self, x, steps=1, **kwargs):

        num_epochs = int(1 + (self.max_num_epochs - 1) * steps)

        train = np.concatenate((self.train, self.valid))
        train_targets = np.concatenate((self.train_targets, self.valid_targets))
        lc_curve, cost_curve, train_loss, valid_loss = self.train_net(train, train_targets,
                                              self.test, self.test_targets,
                                              init_learning_rate=np.float32(np.power(10., x[0])),
                                              l2_reg=np.float32(np.power(10., x[1])),
                                              batch_size=np.int32(x[2]),
                                              gamma=np.float32(np.power(10, x[3])),
                                              power=np.float32(x[4]),
                                              momentum=np.float32(x[5]),
                                              n_units_1=np.int32(np.power(2, x[6])),
                                              n_units_2=np.int32(np.power(2, x[7])),
                                              dropout_rate_1=np.float32(x[8]),
                                              dropout_rate_2=np.float32(x[9]),
                                              num_epochs=np.int32(num_epochs))
        y = lc_curve[-1]
        c = cost_curve[-1]
        return {'function_value': y, "cost": c, "learning_curve": lc_curve}

    @staticmethod
    def get_configuration_space():
        cs = CS.ConfigurationSpace(seed=np.random.randint(1, 100000))
        cs.generate_all_continuous_from_bounds(FullyConnectedNetwork.
                                               get_meta_information()['bounds'])
        return cs

    @staticmethod
    def get_meta_information():
        return {'name': 'Fully Connected Network',
                'bounds': [[-6, 0],  # init_learning_rate
                           [-8, -1],  # l2_reg
                           [32, 512],  # batch_size
                           [-3, -1],  # gamma
                           [0, 1],  # power
                           [0.3, 0.999],  # momentum
                           [5, 12],  # n_units_1
                           [5, 12],  # n_units_2
                           [0.0, 0.99],  # dropout_rate_1
                           [0.0, 0.99]],  # dropout_rate_2
                'references': ["@article{klein-bnn16a,"
                               "author = {A. Klein and S. Falkner and T. Springenberg and F. Hutter},"
                               "title = {Bayesian Neural Network for Predicting Learning Curves},"
                               "booktitle = {NIPS 2016 Bayesian Neural Network Workshop},"
                               "month = dec,"
                               "year = {2016}}"]
                }

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

    def train_net(self, train, train_targets,
                  valid, valid_targets,
                  init_learning_rate, l2_reg,
                  batch_size, gamma, power,
                  momentum, n_units_1, n_units_2,
                  dropout_rate_1, dropout_rate_2,
                  num_epochs):

        start_time = time.time()

        input_var = T.fmatrix('inputs')
        target_var = T.ivector('targets')

        # Build net
        network = lasagne.layers.InputLayer(shape=(None, 28 * 28),
                                            input_var=input_var)

        network = lasagne.layers.batch_norm(lasagne.layers.DenseLayer(
            network,
            num_units=n_units_1,
            W=lasagne.init.HeNormal(),
            b=lasagne.init.Constant(val=0.0),
            nonlinearity=lasagne.nonlinearities.rectify))

        network = lasagne.layers.DropoutLayer(network, p=dropout_rate_1)

        network = lasagne.layers.batch_norm(lasagne.layers.DenseLayer(
            network,
            num_units=n_units_2,
            W=lasagne.init.HeNormal(),
            b=lasagne.init.Constant(val=0.0),
            nonlinearity=lasagne.nonlinearities.rectify))

        network = lasagne.layers.DropoutLayer(network, p=dropout_rate_2)

        network = lasagne.layers.\
            DenseLayer(network, num_units=self.num_classes,
                       nonlinearity=lasagne.nonlinearities.softmax)

        # Define Theano functions
        params = lasagne.layers.get_all_params(network, trainable=True)
        prediction = lasagne.layers.get_output(network)
        loss = lasagne.objectives.categorical_crossentropy(prediction,
                                                           target_var)
        # Add l2 regularization for the weights
        l2_penalty = l2_reg * lasagne.regularization.\
            regularize_network_params(network, lasagne.regularization.l2)
        loss += l2_penalty
        loss = loss.mean()

        test_prediction = lasagne.layers.get_output(network, deterministic=True)
        test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                                target_var)
        test_loss = test_loss.mean()

        test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                          dtype=theano.config.floatX)

        learning_rate = theano.shared(init_learning_rate)
        epoch = T.fscalar("epoch")
        inv_policy = (1 + gamma * epoch) ** (-power)

        adapt_lr = theano.function([epoch], learning_rate, updates=[(learning_rate, init_learning_rate * inv_policy)])

        updates = lasagne.updates.momentum(loss, params,
                                           learning_rate=learning_rate,
                                           momentum=momentum)

        train_fn = theano.function([input_var, target_var], loss, updates=updates)

        val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

        print("Starting training...")

        learning_curve = np.zeros([num_epochs])
        train_loss = np.zeros([num_epochs])
        valid_loss = np.zeros([num_epochs])
        cost = np.zeros([num_epochs])

        for e in range(num_epochs):

            epoch_start_time = time.time()
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

            print("Epoch {} of {} took {:.3f}s".format(e + 1, num_epochs, time.time() - epoch_start_time))
            print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
            print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
            print("  validation accuracy:\t\t{:.2f} %".format(val_acc / val_batches * 100))

            learning_curve[e] = 1 - val_acc / val_batches
            train_loss[e] = train_err / train_batches
            valid_loss[e] = val_err / val_batches
            cost[e] = time.time() - start_time

            adapt_lr(e + 1)

        return learning_curve, cost, train_loss, valid_loss


class FCNetOnMnist(FullyConnectedNetwork):

    def get_data(self):
        # This function loads the MNIST data, it's copied from the Lasagne
        # tutorial. We first define a download function, supporting both
        # Python 2 and 3.

        if sys.version_info[0] == 2:
            from urllib import urlretrieve
        else:
            from urllib.request import urlretrieve

        def download(filename, save_to,
                     source='http://yann.lecun.com/exdb/mnist/'):
            print("Downloading %s" % filename)
            urlretrieve(source + filename, save_to)

        # We then define functions for loading MNIST images and labels.
        # For convenience, they also download the requested files if needed.
        import gzip

        def load_mnist_images(filename, save_to):
            save_fl = os.path.join(save_to, filename)

            if not os.path.exists(save_fl):
                download(filename=filename, save_to=save_fl)

            # Read the inputs in Yann LeCun's binary format.
            with gzip.open(save_fl, 'rb') as f:
                data = np.frombuffer(f.read(), np.uint8, offset=16)
            # The inputs are vectors now, we reshape them to monochrome 2D
            # images, following the shape convention:
            # (examples, channels, rows, columns)
            data = data.reshape(-1, 1, 28, 28)
            # The inputs come as bytes, we convert them to float32 in range
            # [0,1]. (Actually to range [0, 255/256], for compatibility to the
            # version provided at:
            #  http://deeplearning.net/data/mnist/mnist.pkl.gz.)
            return data / np.float32(256)

        def load_mnist_labels(filename, save_to):
            save_fl = os.path.join(os.path.join(save_to, filename))

            if not os.path.exists(save_fl):
                download(filename=filename, save_to=save_fl)

            # Read the labels in Yann LeCun's binary format.
            with gzip.open(save_fl, 'rb') as f:
                data = np.frombuffer(f.read(), np.uint8, offset=8)
            # Labels are vectors of integers now, that's exactly what we want.
            return data

        if not os.path.isdir(self.path):
                os.makedirs(self.path)

        # We can now download and read the training and test set images and
        # labels.
        X_train = load_mnist_images(filename='train-images-idx3-ubyte.gz',
                                    save_to=self.path)
        y_train = load_mnist_labels(filename='train-labels-idx1-ubyte.gz',
                                    save_to=self.path)
        X_test = load_mnist_images(filename='t10k-images-idx3-ubyte.gz',
                                   save_to=self.path)
        y_test = load_mnist_labels(filename='t10k-labels-idx1-ubyte.gz',
                                   save_to=self.path)

        # We reserve the last 10000 training examples for validation.
        X_train, X_val = X_train[:-10000], X_train[-10000:]
        y_train, y_val = y_train[:-10000], y_train[-10000:]

        X_train = X_train.reshape(X_train.shape[0], 28 * 28)
        X_val = X_val.reshape(X_val.shape[0], 28 * 28)
        X_test = X_test.reshape(X_test.shape[0], 28 * 28)

        # We just return all the arrays in order, as expected in main().
        # (It doesn't matter how we do this as long as we can read them again.)
        return X_train, y_train, X_val, y_val, X_test, y_test
