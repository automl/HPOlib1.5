import abc
import gzip
import logging
import pickle
import os
import tarfile
import zipfile

from urllib.request import urlretrieve

import numpy as np
from scipy.io import loadmat

import hpolib


class DataManager(object, metaclass=abc.ABCMeta):

    def __init__(self):

        self.logger = logging.getLogger("DataManager")

    @abc.abstractmethod
    def load(self):
        """
        Loads data from data directory as defined in _config.data_directory
        """
        raise NotImplementedError()


class HoldoutDataManager(DataManager):

    def __init__(self):

        super().__init__()

        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None


class CrossvalidationDataManager(DataManager):

    def __init__(self):

        super().__init__()

        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None


class MNISTData(HoldoutDataManager):

    def __init__(self):
        self.url_source = 'http://yann.lecun.com/exdb/mnist/'
        self.save_to = os.path.join(hpolib._config.data_dir, "MNIST")

        if not os.path.isdir(self.save_to):
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

        # Split data in training and validation
        X_train, X_val = X_train[:-10000], X_train[-10000:]
        y_train, y_val = y_train[:-10000], y_train[-10000:]

        assert X_train.shape[0] == 50000, X_train.shape
        assert X_val.shape[0] == 10000, X_val.shape
        assert X_test.shape[0] == 10000, X_test.shape

        # Reshape data to NxD
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

class MNISTDataCrossvalidation(MNISTData, CrossvalidationDataManager):

    def load(self):
        X_train, y_train, X_val, y_val, X_test, y_test = \
            super(MNISTDataCrossvalidation, self).load()
        X_train = np.concatenate([X_train, X_val], axis=0)
        y_train = np.concatenate([y_train, y_val], axis=0)
        return X_train, y_train, X_test, y_test


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

        xs = []
        ys = []
        for j in range(5):
            fh = open(self.__load_data(filename='data_batch_%d' % (j + 1)),
                      "rb")
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

        # Subtract per-pixel mean
        pixel_mean = np.mean(x[0:50000], axis=0)

        x -= pixel_mean

        # Split in training, validation and test
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

        Returns
        -------
        filename: string
        """

        save_fl = os.path.join(self.save_to, "cifar-10-batches-py", filename)
        if not os.path.exists(save_fl):
            self.logger.debug("Downloading %s to %s",
                              self.url_source, save_fl)
            urlretrieve(self.url_source,
                        self.save_to + "cifar-10-python.tar.gz")
            tar = tarfile.open(self.save_to + "cifar-10-python.tar.gz")
            tar.extractall(self.save_to)

        else:
            self.logger.debug("Load data %s", save_fl)

        return save_fl


class SVHNData(DataManager):

    def __init__(self):
        self.url_source = 'http://ufldl.stanford.edu/housenumbers/'
        self.logger = logging.getLogger("DataManager")
        self.save_to = os.path.join(hpolib._config.data_dir, "svhn/")

        self.n_train_all = 73257
        self.n_valid = 6000
        self.n_train = self.n_train_all - self.n_valid
        self.n_test = 26032

        if not os.path.isdir(self.save_to):
            self.logger.debug("Create directory %s", self.save_to)
            os.makedirs(self.save_to)

        super(SVHNData, self).__init__()

    def load(self):
        """
        Loads SVHN from data directory as defined in _config.data_directory.
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
        X, y, X_test, y_test = self.__load_data("train_32x32.mat", "test_32x32.mat")

        # Change the label encoding from [1, ... 10] to [0, ..., 9]
        y[y == 10] = 0
        y_test[y_test == 10] = 0

        X_train = X[:self.n_train, :, :, :]
        y_train = y[:self.n_train]
        X_valid = X[self.n_train:self.n_train_all, :, :, :]
        y_valid = y[self.n_train:self.n_train_all]

        X_train = np.array(X_train, dtype=np.float32)
        X_valid = np.array(X_valid, dtype=np.float32)
        X_test = np.array(X_test, dtype=np.float32)

        all_X = [X_train, X_valid, X_test]

        # Subtract per pixel mean
        for X in all_X:
            data_shape = X.shape
            X = X.reshape(X.shape[0], np.product(X.shape[1:]))
            X -= X.mean(axis=1)[:, np.newaxis]
            X = X.reshape(data_shape)

        return X_train, y_train[:, 0], X_valid, y_valid[:, 0], X_test, y_test[:, 0]

    def __load_data(self, filename_train, filename_test):
        """
        Loads data in binary format as available under 'http://ufldl.stanford.edu/housenumbers/'.

        Parameters
        ----------
        filename_train: str
            file to download
        filename_test: str
            file to download

        Returns
        -------
        filename: string
        """
        save_fl = os.path.join(self.save_to, filename_train)
        if not os.path.exists(save_fl):
            self.logger.debug("Downloading %s to %s",
                              self.url_source + filename_train, save_fl)
            urlretrieve(self.url_source + filename_train, self.save_to + filename_train)

        else:
            self.logger.debug("Load data %s", save_fl)

        train_data = loadmat(save_fl)

        X_train = train_data['X'].T
        y_train = train_data['y']

        save_fl = os.path.join(self.save_to, filename_test)
        if not os.path.exists(save_fl):
            self.logger.debug("Downloading %s to %s",
                              self.url_source + filename_test, save_fl)
            urlretrieve(self.url_source + filename_test, self.save_to + filename_test)

        else:
            self.logger.debug("Load data %s", save_fl)

        test_data = loadmat(save_fl)

        X_test = test_data['X'].T
        y_test = test_data['y']

        return X_train, y_train, X_test, y_test


class BostonHousingData(HoldoutDataManager):

    def __init__(self):
        super(BostonHousingData, self).__init__()
        self.url_source = 'https://archive.ics.uci.edu/ml/machine-learning-databases/housing/'
        self.logger = logging.getLogger("DataManager")
        self.save_to = os.path.join(hpolib._config.data_dir, "BostonHousing")

        if not os.path.isdir(self.save_to):
            self.logger.debug("Create directory %s", self.save_to)
            os.makedirs(self.save_to)

    def load(self):
        """
        Loads BostonHousing from data directory as defined in _config.data_directory.
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
        
        data = self.__load_data('housing.data')
        
        N = data.shape[0]
        
        n_train = int(N * 0.6)
        n_val = int(N * 0.2)
        
        X_train, y_train = data[:n_train, :-1], data[:n_train,-1]
        X_val, y_val = data[n_train:n_train+n_val, :-1], data[n_train:n_train+n_val,-1]
        X_test, y_test = data[n_train+n_val:, :-1], data[n_train+n_val:,-1]
        
        return X_train, y_train, X_val, y_val, X_test, y_test

    def __load_data(self, filename, images=False):
        """
        Loads data from UCI website
        https://archive.ics.uci.edu/ml/machine-learning-databases/housing/
        If necessary downloads data, otherwise loads data from data_directory

        Parameters
        ----------
        filename: str
            file to download
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

        return(np.loadtxt(save_fl))


class ProteinStructureData(HoldoutDataManager):

    def __init__(self):
        super(ProteinStructureData, self).__init__()
        self.url_source = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00265/'
        self.logger = logging.getLogger("DataManager")
        self.save_to = os.path.join(hpolib._config.data_dir, "ProteinStructure")

        if not os.path.isdir(self.save_to):
            self.logger.debug("Create directory %s", self.save_to)
            os.makedirs(self.save_to)



    def load(self):
        """
        Loads Physicochemical Properties of Protein Tertiary Structure Data Set
        from data directory as defined in _config.data_directory.
        Downloads data if necessary from UCI.

        Returns
        -------
        X_train: np.array
        y_train: np.array
        X_val: np.array
        y_val: np.array
        X_test: np.array
        y_test: np.array
        """
        
        data = self.__load_data('CASP.csv')
        
        N = data.shape[0]
        
        n_train = int(N * 0.6)
        n_val = int(N * 0.2)
        
        # note the target value is the first column for this dataset!
        X_train, y_train = data[:n_train, 1:], data[:n_train,0]
        X_val, y_val = data[n_train:n_train+n_val, 1:], data[n_train:n_train+n_val,0]
        X_test, y_test = data[n_train+n_val:, 1:], data[n_train+n_val:,0]
        
        return X_train, y_train, X_val, y_val, X_test, y_test

    def __load_data(self, filename, images=False):
        """
        Loads data from UCI website
        https://archive.ics.uci.edu/ml/machine-learning-databases/housing/
        If necessary downloads data, otherwise loads data from data_directory

        Parameters
        ----------
        filename: str
            file to download
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

        return(np.loadtxt(save_fl, delimiter=',', skiprows=1))




class YearPredictionMSDData(HoldoutDataManager):

    def __init__(self):
        super(YearPredictionMSDData, self).__init__()
        self.url_source = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00203/'
        self.logger = logging.getLogger("DataManager")
        self.save_to = os.path.join(hpolib._config.data_dir, "YearPredictionMSD")

        if not os.path.isdir(self.save_to):
            self.logger.debug("Create directory %s", self.save_to)
            os.makedirs(self.save_to)



    def load(self):
        """
        Loads Physicochemical Properties of Protein Tertiary Structure Data Set
        from data directory as defined in _config.data_directory.
        Downloads data if necessary from UCI.

        Returns
        -------
        X_train: np.array
        y_train: np.array
        X_val: np.array
        y_val: np.array
        X_test: np.array
        y_test: np.array
        """
        
        data = self.__load_data()
        
        n_trn = int(data.shape[0]* 0.7)
        n_val   = int(data.shape[0]* 0.2)
        
        # note the target value is the first column for this dataset!
        X_trn, y_trn = data[           :n_trn,       1:], data[:n_trn,0]
        X_val, y_val = data[n_trn      :n_trn+n_val, 1:], data[n_trn:n_trn+n_val,0]
        X_tst, y_tst = data[n_trn+n_val:,            1:], data[n_trn+n_val:,0]
        
        return X_trn, y_trn, X_val, y_val, X_tst, y_tst

    def __load_data(self):
        """
        Loads data from UCI website
        https://archive.ics.uci.edu/ml/machine-learning-databases/00203/
        If necessary downloads data, otherwise loads data from data_directory

        Parameters
        ----------
        filename: str
            file to download
        Returns
        -------
        data: array
        """

        data_fn = os.path.join(self.save_to, 'data.npy')

        if not os.path.exists(data_fn):
            
            orig_data_fn = os.path.join(self.save_to, 'YearPredictionMSD.txt.zip')
            if not os.path.exists(orig_data_fn):
            
                self.logger.debug("Downloading %s to %s",
                              self.url_source + 'YearPredictionMSD.txt.zip', orig_data_fn)
            
                urlretrieve(self.url_source + 'YearPredictionMSD.txt.zip', orig_data_fn)

            with zipfile.ZipFile(orig_data_fn, 'r') as zf:
                with zf.open('YearPredictionMSD.txt','r') as fh:
                    data = np.loadtxt(fh, delimiter=',')        
        
            np.save(data_fn, data)
        else:
            data = np.load(data_fn)

        return(data)

