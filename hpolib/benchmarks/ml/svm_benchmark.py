import time

import os
import numpy as np
import ConfigSpace as CS

from scipy import sparse
from sklearn import svm
from sklearn.model_selection import StratifiedShuffleSplit


from hpolib.abstract_benchmark import AbstractBenchmark
from hpolib.util.data_manager import MNISTData
from hpolib.util.openml_data_manager import OpenMLHoldoutDataManager
import hpolib.util.rng_helper as rng_helper


class SupportVectorMachine(AbstractBenchmark):
    """
        Hyperparameter optimization task to optimize the regularization
        parameter C and the kernel parameter gamma of a support vector machine.
        Both hyperparameters are optimized on a log scale in [-10, 10].

        The test data set is only used for a final offline evaluation of
        a configuration. For that the validation and training data is
        concatenated to form the whole training data set.
    """
    def __init__(self, rng=None, cache_size=2000):
        """

        Parameters
        ----------
        rng: int/None/RandomState
            set up rng
        cache_size: int
            kernel cache size (in MB) for the SVM
        """
        self.train, self.train_targets, self.valid, self.valid_targets, \
            self.test, self.test_targets = self.get_data()

        # Use 10 time the number of classes as lower bound for the dataset
        # fraction
        n_classes = np.unique(self.train_targets).shape[0]
        self.cache_size = cache_size
        if os.environ.get("TRAVIS", 0) == "true":
            # If run on travis.ci, don't use higher cache size
            self.cache_size = 200
        self.s_min = float(10 * n_classes) / self.train.shape[0]

        super(SupportVectorMachine, self).__init__()

        self.n_calls = 0
        self.rng = rng_helper.create_rng(rng)

    def get_data(self):
        raise NotImplementedError()

    @AbstractBenchmark._check_configuration
    @AbstractBenchmark._configuration_as_array
    def objective_function(self, x, dataset_fraction=1, **kwargs):
        start_time = time.time()

        rng = kwargs.get("rng", None)
        self.rng = rng_helper.get_rng(rng=rng, self_rng=self.rng)

        # Stratified shuffle training data
        if np.round(dataset_fraction, 3) < 1.0:
            sss = StratifiedShuffleSplit(n_splits=1, train_size=np.round(dataset_fraction, 3), test_size=None)
            idx = list(sss.split(self.train, self.train_targets))[0][0]

            train = self.train[idx]
            train_targets = self.train_targets[idx]
        else:
            train = self.train
            train_targets = self.train_targets

        # Transform hyperparameters to linear scale
        C = np.exp(float(x[0]))
        gamma = np.exp(float(x[1]))

        # Train support vector machine
        clf = svm.SVC(gamma=gamma, C=C, random_state=self.rng, cache_size=self.cache_size)
        clf.fit(train, train_targets)

        # Compute validation error
        y = 1 - clf.score(self.valid, self.valid_targets)
        c = time.time() - start_time

        return {'function_value': y, "cost": c}
    
    @AbstractBenchmark._check_configuration
    @AbstractBenchmark._configuration_as_array
    def objective_function_test(self, x, **kwargs):
        start_time = time.time()

        rng = kwargs.get("rng", None)
        self.rng = rng_helper.get_rng(rng=rng, self_rng=self.rng)

        # Concatenate training and validation dataset
        if type(self.train) == sparse.csr.csr_matrix or type(self.valid) == sparse.csr.csr_matrix:
            train = sparse.vstack((self.train, self.valid))
        else:
            train = np.concatenate((self.train, self.valid))

        train_targets = np.concatenate((self.train_targets, self.valid_targets))

        # Transform hyperparameters to linear scale
        C = np.exp(float(x[0]))
        gamma = np.exp(float(x[1]))

        # Train support vector machine
        clf = svm.SVC(gamma=gamma, C=C, random_state=self.rng, cache_size=self.cache_size)
        clf.fit(train, train_targets)

        # Compute test error
        y = 1 - clf.score(self.test, self.test_targets)
        c = time.time() - start_time

        return {'function_value': y, "cost": c}

    @staticmethod
    def get_configuration_space():
        cs = CS.ConfigurationSpace()
        cs.generate_all_continuous_from_bounds(SupportVectorMachine.get_meta_information()['bounds'])
        return cs

    @staticmethod
    def get_meta_information():
        return {'name': 'Support Vector Machine',
                'bounds': [[-10.0, 10.0],  # C
                           [-10.0, 10.0]],  # gamma
                # as defined in https://github.com/automl/RoBO/blob/master/experiments/fabolas/run_bo.py#L24
                'num_function_evals': 15,
                'references': ["@InProceedings{klein-aistats17,"
                               "author = {A. Klein and S. Falkner and S. Bartels and P. Hennig and F. Hutter},"
                               "title = {Fast {Bayesian} Optimization of Machine"
                               "Learning Hyperparameters on Large Datasets},"
                               "booktitle = {Proceedings of the AISTATS conference},"
                               "year = {2017}}"]
                }


class SvmOnMnist(SupportVectorMachine):

    def get_data(self):
        dm = MNISTData()
        return dm.load()

    @staticmethod
    def get_meta_information():
        d = SupportVectorMachine.get_meta_information()
        dataset_ref = ["@article{lecun-ieee98,"
                       "title={Gradient-based learning applied to document recognition},"
                       "author={Y. LeCun and L. Bottou and Y. Bengio and P. Haffner},"
                       "journal={Proceedings of the IEEE},"
                       "pages={2278--2324},"
                       "year={1998},"
                       "publisher={IEEE}"]
        d["references"].append(dataset_ref)
        return d


class SvmOnVehicle(SupportVectorMachine):

    def get_data(self):
        dm = OpenMLHoldoutDataManager(openml_task_id=75191)
        return dm.load()


class SvmOnCovertype(SupportVectorMachine):

    def get_data(self):
        dm = OpenMLHoldoutDataManager(openml_task_id=2118)
        return dm.load()


class SvmOnLetter(SupportVectorMachine):

    def get_data(self):
        dm = OpenMLHoldoutDataManager(openml_task_id=236)
        return dm.load()


class SvmOnAdult(SupportVectorMachine):

    def get_data(self):
        dm = OpenMLHoldoutDataManager(openml_task_id=2117)
        X_train, y_train, X_val, y_val, X_test, y_test = dm.load()
        from sklearn.preprocessing import Imputer
        imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
        imp.fit(X_train)
        X_train = imp.transform(X_train)
        X_val = imp.transform(X_val)
        X_test = imp.transform(X_test)
        return X_train, y_train, X_val, y_val, X_test, y_test


class SvmOnHiggs(SupportVectorMachine):

    def get_data(self):
        dm = OpenMLHoldoutDataManager(openml_task_id=75101)
        X_train, y_train, X_val, y_val, X_test, y_test = dm.load()
        from sklearn.preprocessing import Imputer
        imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
        imp.fit(X_train)
        X_train = imp.transform(X_train)
        X_val = imp.transform(X_val)
        X_test = imp.transform(X_test)
        return X_train, y_train, X_val, y_val, X_test, y_test
