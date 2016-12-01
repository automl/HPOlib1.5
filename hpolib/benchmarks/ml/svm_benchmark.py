import os
import sys
import time

import numpy as np
from sklearn import svm

import ConfigSpace as CS

from hpolib.abstract_benchmark import AbstractBenchmark


class SupportVectorMachine(AbstractBenchmark):
    """
        Hyperparameter optimization task to optimize the regularization
        parameter C and the kernel parameter gamma of a support vector machine.
        Both hyperparameters are optimized on a log scale in [-10, 10].

        The test data set is only used for a final offline evaluation of
        a configuration. For that the validation and training data is
        concatenated to form the whole training data set.
    """
    def __init__(self, path=None):
        """

        Parameters
        ----------
        path: str
            directory to find or download dataset to
        """

        self.train, self.train_targets, self.valid, self.valid_targets, \
            self.test, self.test_targets = self.get_data(path)

        # Use 10 time the number of classes as lower bound for the dataset
        # fraction
        n_classes = np.unique(self.train_targets).shape[0]
        self.s_min = float(10 * n_classes) / self.train.shape[0]

        super(SupportVectorMachine, self).__init__()

        self.n_calls = 0

    def get_data(self, path):
        pass

    @AbstractBenchmark._check_configuration
    @AbstractBenchmark._configuration_as_array
    def objective_function(self, x, dataset_fraction=1, **kwargs):
        start_time = time.time()

        # Shuffle training data
        shuffle = np.random.permutation(self.train.shape[0])
        size = int(dataset_fraction * self.train.shape[0])

        # Split of dataset subset
        train = self.train[shuffle[:size]]
        train_targets = self.train_targets[shuffle[:size]]

        # Transform hyperparameters to linear scale
        C = np.exp(float(x[0]))
        gamma = np.exp(float(x[1]))

        # Train support vector machine
        clf = svm.SVC(gamma=gamma, C=C)
        clf.fit(train, train_targets)

        # Compute validation error
        y = 1 - clf.score(self.valid, self.valid_targets)
        c = time.time() - start_time

        return {'function_value': y, "cost": c}
    
    @AbstractBenchmark._check_configuration
    @AbstractBenchmark._configuration_as_array
    def objective_function_test(self, x, **kwargs):
        start_time = time.time()

        # Concatenate training and validation dataset
        train = np.concatenate((self.train, self.valid))
        train_targets = np.concatenate((self.train_targets, self.valid_targets))

        # Transform hyperparameters to linear scale
        C = np.exp(float(x[0]))
        gamma = np.exp(float(x[1]))

        # Train support vector machine
        clf = svm.SVC(gamma=gamma, C=C)
        clf.fit(train, train_targets)

        # Compute test error
        y = 1 - clf.score(self.test, self.test_targets)
        c = time.time() - start_time

        return {'function_value': y, "cost": c}

    @staticmethod
    def get_configuration_space():
        cs = CS.ConfigurationSpace(seed=np.random.randint(1, 100000))
        cs.generate_all_continuous_from_bounds(SupportVectorMachine.
                                               get_meta_information()['bounds'])
        return cs

    @staticmethod
    def get_meta_information():
        return {'name': 'Support Vector Machine',
                'bounds': [[-10, 10],  # C
                           [-10, 10]]  # gamma
                }

class SvmOnMnist(SupportVectorMachine):

    def get_data(self, path):
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
            #  (examples, channels, rows, columns)
            data = data.reshape(-1, 1, 28, 28)
            # The inputs come as bytes, we convert them to float32 in range
            # [0,1]. (Actually to range [0, 255/256], for compatibility to the
            # version provided at:
            #   http://deeplearning.net/data/mnist/mnist.pkl.gz.)
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

        if not os.path.isdir(path):
                os.makedirs(path)

        # Download and read the training and test set images and labels.
        X_train = load_mnist_images(filename='train-images-idx3-ubyte.gz',
                                    save_to=path)
        y_train = load_mnist_labels(filename='train-labels-idx1-ubyte.gz',
                                    save_to=path)
        X_test = load_mnist_images(filename='t10k-images-idx3-ubyte.gz',
                                   save_to=path)
        y_test = load_mnist_labels(filename='t10k-labels-idx1-ubyte.gz',
                                   save_to=path)

        # We reserve the last 10000 training examples for validation.
        X_train, X_val = X_train[:-10000], X_train[-10000:]
        y_train, y_val = y_train[:-10000], y_train[-10000:]

        X_train = X_train.reshape(X_train.shape[0], 28 * 28)
        X_val = X_val.reshape(X_val.shape[0], 28 * 28)
        X_test = X_test.reshape(X_test.shape[0], 28 * 28)

        # We just return all the arrays in order, as expected in main().
        # (It doesn't matter how we do this as long as we can read them again.)
        return X_train, y_train, X_val, y_val, X_test, y_test