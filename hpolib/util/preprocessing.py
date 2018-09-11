import numpy as np

from sklearn.preprocessing import OneHotEncoder, StandardScaler, Imputer


class DataDetergent(object):

    def __init__(self,
                 one_hot_encoding=True,
                 remove_constant_features=True,
                 normalize=True,
                 categorical_features=None,
                 impute_nans=False,
                 dtype=None):
        """
        Simple data preprocessor loosely following the sklearn API

        Parameters:
        -----------
        one_hot_encoding: bool
            whether to perform 1-hot-encoding

        remove_constant_features: bool
            whether to remove constant features from the data

        normalize: bool
            whether to scale the values to zero mean and unit variance

        categorical_features: np.array(dtype=np.bool)
            array indicating which features are categoricals
            only used if one_hot_encoding is true

        impute_nans: bool
            whether to encode nans to the mean value

        dtype: numpy datatype
            datatype of the array after the one hot encoding
            maybe interesting for running it on GPUs with float32
        """

        if one_hot_encoding and categorical_features is not None:
            dtype = np.float if dtype is None else dtype
            self.OHE = OneHotEncoder(n_values='auto',
                                     categorical_features=categorical_features,
                                     sparse=False,
                                     dtype=dtype)
            self.categorical_features = np.array(categorical_features)
        else:
            self.OHE = None
            self.categorical_features = None

        if normalize:
            self.scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
        else:
            self.scaler = None

        if impute_nans:
            self.imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
        else:
            self.imputer = None

        self.remove_constant_features = remove_constant_features

    def fit_transform(self, X):
        """
            fit the preprocessor to the data and transform it at the same time

            Parameters:
            -----------

            X: numpy.array
                the data to be transformed

            Returns:
            --------
            numpy.array
                the transformed data
        """

        if self.remove_constant_features:
            self.active_indices = np.nonzero(np.var(X, axis=0))[0]
            if self.categorical_features is not None:
                self.OHE.set_params(categorical_features=self.categorical_features[self.active_indices])
        else:
            self.active_indices = np.ones(X.shape[1], dtype=np.bool)

        _X = X[:, self.active_indices]

        if self.imputer is not None:
            _X = self.imputer.fit_transform(_X)

        if self.OHE is not None:
            _X = self.OHE.fit_transform(_X)

        if self.scaler is not None:
            _X = self.scaler.fit_transform(_X)

        return _X

    def transform(self, X):
        """
            transform new data according to the training data

            Parameters:
            -----------

            X: numpy.array
                the data to be transformed

            Returns:
            --------
            numpy.array
                the transformed data
        """
        _X = X[:, self.active_indices]

        if self.imputer is not None:
            _X = self.imputer.fit_transform(_X)

        if self.OHE is not None:
            _X = self.OHE.transform(_X)

        if self.scaler is not None:
            _X = self.scaler.transform(_X)

        return _X
