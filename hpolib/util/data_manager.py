import abc
import gzip
import logging
import os
from urllib.request import urlretrieve

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

        pass

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
            self.logger.debug("Create directory %s" % self.save_to)
            os.mkdir(self.save_to)

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

        assert X_train.shape[0] == 50000
        assert X_val.shape[0] == 10000
        assert X_test.shape[0] == 10000

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
            self.logger.debug("Downloading %s to %s" %
                              (self.url_source + filename, save_fl))
            urlretrieve(self.url_source + filename, save_fl)
        else:
            self.logger.debug("Load data %s" % save_fl)

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


