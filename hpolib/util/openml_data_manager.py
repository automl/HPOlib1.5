import logging
import os

import numpy as np
import openml
from sklearn.model_selection import train_test_split

import hpolib
from hpolib.util.data_manager import DataManager


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

        openml.config.apikey = '953f6621518c13791dbbfc6d3698f5ad'
        openml.config.cachedir = self.save_to

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