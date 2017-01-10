import abc
import gzip
import logging
import openml
import pickle
import os
import tarfile

from urllib.request import urlretrieve
from sklearn.cross_validation import train_test_split

import numpy as np

import hpolib


class DataManager(object, metaclass=abc.ABCMeta):

    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None

        self.logger = logging.getLogger("DataManager")

    @abc.abstractmethod
    def load(self):
        """
        Loads data from data directory as defined in _config.data_directory
        """
        raise NotImplementedError()


class MNISTData(DataManager):

    def __init__(self):
        self.url_source = 'http://yann.lecun.com/exdb/mnist/'
        self.logger = logging.getLogger("DataManager")
        self.save_to = os.path.join(hpolib._config.data_dir, "MNIST")

        if not os.path.isdir(self.save_to):
            self.logger.debug("Create directory %s", self.save_to)
            os.makedirs(self.save_to)

        super(MNISTData, self).__init__()

    def load(self):
        """
        Loads MNIST from data directory as defined in _config.data_directory.
        Downloads data if necessary. Code is copied and modified from the
        Lasagne tutorial.

        Returns
        -------
        X_train: np.array
        y_train: np.array
        X_val: np.array
        y_val: np.array
        X_test: np.array
        y_test: np.array
        """
        X_train = self.__load_data(filename='train-images-idx3-ubyte.gz',
                                   images=True)
        y_train = self.__load_data(filename='train-labels-idx1-ubyte.gz')
        X_test = self.__load_data(filename='t10k-images-idx3-ubyte.gz',
                                  images=True)
        y_test = self.__load_data(filename='t10k-labels-idx1-ubyte.gz')

        # Split data
        X_train, X_val = X_train[:-10000], X_train[-10000:]
        y_train, y_val = y_train[:-10000], y_train[-10000:]

        assert X_train.shape[0] == 50000, X_train.shape
        assert X_val.shape[0] == 10000, X_val.shape
        assert X_test.shape[0] == 10000, X_test.shape

        # Reshape data
        X_train = X_train.reshape(X_train.shape[0], 28 * 28)
        X_val = X_val.reshape(X_val.shape[0], 28 * 28)
        X_test = X_test.reshape(X_test.shape[0], 28 * 28)

        return X_train, y_train, X_val, y_val, X_test, y_test

    def __load_data(self, filename, images=False):
        """
        Loads data in Yann LeCun's binary format as available under
        'http://yann.lecun.com/exdb/mnist/'.
        If necessary downloads data, otherwise loads data from data_directory

        Parameters
        ----------
        filename: str
            file to download
        save_to: str
            directory to store file
        images: bool
            if True converts data to X

        Returns
        -------
        data: array
        """

        # 1) If necessary download data
        save_fl = os.path.join(self.save_to, filename)
        if not os.path.exists(save_fl):
            self.logger.debug("Downloading %s to %s",
                              self.url_source + filename, save_fl)
            urlretrieve(self.url_source + filename, save_fl)
        else:
            self.logger.debug("Load data %s", save_fl)

        # 2) Read in data
        if images:
            with gzip.open(save_fl, 'rb') as f:
                data = np.frombuffer(f.read(), np.uint8, offset=16)

            # Follow the shape convention: (examples, channels, rows, columns)
            data = data.reshape(-1, 1, 28, 28)
            # Convert them to float32 in range [0,1].
            # (Actually to range [0, 255/256], for compatibility to the version
            # provided at: http://deeplearning.net/data/mnist/mnist.pkl.gz.
            data = data / np.float32(256)
        else:
            with gzip.open(save_fl, 'rb') as f:
                data = np.frombuffer(f.read(), np.uint8, offset=8)

        return data


class OpenMLData(DataManager):

    def __init__(self, openml_task_id, rng=None):
        self.logger = logging.getLogger("DataManager")
        self.save_to = os.path.join(hpolib._config.data_dir, "OpenML")
        self.task_id = openml_task_id

        if rng is None:
            self.rng = np.random.RandomState()
        else:
            self.rng = rng

        if not os.path.isdir(self.save_to):
            self.logger.debug("Create directory %s", self.save_to)
            os.makedirs(self.save_to)

        super(OpenMLData, self).__init__()

    def load(self):
        """
        Loads dataset from OpenML in _config.data_directory.
        Downloads data if necessary.

        Returns
        -------
        X_train: np.array
        y_train: np.array
        X_val: np.array
        y_val: np.array
        X_test: np.array
        y_test: np.array
        """

        task = openml.tasks.get_task(self.task_id)
        try:
            task.get_train_test_split_indices(fold=1, repeat=0)
            raise_exception = True
        except:
            raise_exception = False

        if raise_exception:
            raise ValueError('Task %d has more than one fold. This benchmark '
                             + 'can only work with a single fold.' % self.task_id)

        try:
            task.get_train_test_split_indices(fold=0, repeat=1)
            raise_exception = True
        except:
            raise_exception = False

        if raise_exception:
            raise ValueError('Task %d has more than one repeat. This benchmark '
                             + 'can only work with a single repeat.' % self.task_id)

        train_indices, test_indices = task.get_train_test_split_indices()

        X, y = task.get_X_and_y()

        X_train = X[train_indices]
        y_train = y[train_indices]
        X_test = X[test_indices]
        y_test = y[test_indices]

        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train,
                                                              test_size=0.33,
                                                              random_state=self.rng)

        return X_train, y_train, X_valid, y_valid, X_test, y_test

    def __load_data(self, filename, images=False):
        pass


class CIFAR10Data(DataManager):

    def __init__(self):
        self.url_source = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
        self.logger = logging.getLogger("DataManager")
        self.save_to = os.path.join(hpolib._config.data_dir, "cifar10/")

        if not os.path.isdir(self.save_to):
            self.logger.debug("Create directory %s", self.save_to)
            os.makedirs(self.save_to)

        super(CIFAR10Data, self).__init__()

    def load(self):
        """
        Loads CIFAR10 from data directory as defined in _config.data_directory.
        Downloads data if necessary. Code is copied and modified from the
        Lasagne tutorial.

        Returns
        -------
        X_train: np.array
        y_train: np.array
        X_val: np.array
        y_val: np.array
        X_test: np.array
        y_test: np.array
        """

        xs = []
        ys = []
        for j in range(5):
            fh = open(self.__load_data(filename='data_batch_%d' % (j + 1)), "rb")
            d = pickle.load(fh, encoding='latin1')
            fh.close()
            x = d['data']
            y = d['labels']
            xs.append(x)
            ys.append(y)

        fh = open(self.__load_data(filename='test_batch'), "rb")
        d = pickle.load(fh, encoding='latin1')
        fh.close()

        xs.append(d['data'])
        ys.append(d['labels'])

        x = np.concatenate(xs) / np.float32(255)
        y = np.concatenate(ys)
        x = np.dstack((x[:, :1024], x[:, 1024:2048], x[:, 2048:]))
        x = x.reshape((x.shape[0], 32, 32, 3)).transpose(0, 3, 1, 2)

        # subtract per-pixel mean
        pixel_mean = np.mean(x[0:50000], axis=0)

        x -= pixel_mean

        X_train = x[:40000, :, :, :]
        y_train = y[:40000]

        X_valid = x[40000:50000, :, :, :]
        y_valid = y[40000:50000]

        X_test = x[50000:, :, :, :]
        y_test = y[50000:]

        return X_train, y_train, X_valid, y_valid, X_test, y_test

    def __load_data(self, filename):
        """
        Loads data in binary format as available under 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'.

        Parameters
        ----------
        filename: str
            file to download
        save_to: str
            directory to store file
        images: bool
            if True converts data to X

        Returns
        -------
        filename: string
        """

        save_fl = os.path.join(self.save_to, "cifar-10-batches-py", filename)
        if not os.path.exists(save_fl):
            self.logger.debug("Downloading %s to %s",
                              self.url_source, save_fl)
            urlretrieve(self.url_source, self.save_to + "cifar-10-python.tar.gz")
            tar = tarfile.open(self.save_to + "cifar-10-python.tar.gz")
            tar.extractall(self.save_to)

        else:
            self.logger.debug("Load data %s", save_fl)

        return save_fl
