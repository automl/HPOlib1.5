import time
import numpy as np

from param_net import ParamFCNetClassification
from param_net.util import zero_mean_unit_var_normalization

from hpolib.util import rng_helper
from hpolib.util.data_manager import MNISTData
from hpolib.util.openml_data_manager import OpenMLHoldoutDataManager
from hpolib.abstract_benchmark import AbstractBenchmark

import openml

from sklearn.model_selection import StratifiedShuffleSplit

import keras.backend as K
import tensorflow as T


class ParamNetBenchmark(AbstractBenchmark):

    def __init__(self, n_epochs=100, do_early_stopping=False, rng=None):

        self.train, self.train_targets, self.valid, self.valid_targets, \
            self.test, self.test_targets = self.get_data()
        self.n_epochs = n_epochs
        self.do_early_stopping = do_early_stopping

        # Use 10 time the number of classes as lower bound for the dataset
        # fraction
        self.n_classes = np.unique(self.train_targets).shape[0]
        self.s_min = float(10 * self.n_classes) / self.train.shape[0]
        self.rng = rng_helper.create_rng(rng)

        super(ParamNetBenchmark, self).__init__()

    def get_data(self):
        raise NotImplementedError()

    @AbstractBenchmark._check_configuration
    def objective_function(self, x, dataset_fraction=1, **kwargs):
        start_time = time.time()

        rng = kwargs.get("rng", None)
        self.rng = rng_helper.get_rng(rng=rng, self_rng=self.rng)

        fold = int(float(kwargs.get("fold", 0)))
        folds = kwargs.get('folds', 1)

        if fold == folds:
            # Test fold, run function_test
            return self.objective_function_test(x, rng=rng)

        if dataset_fraction < 1.0:
            sss = StratifiedShuffleSplit(n_splits=1,
                                         train_size=np.round(dataset_fraction, 3),
                                         test_size=None,
                                         random_state=self.rng)
            idx = list(sss.split(self.train, self.train_targets))[0][0]

            train = self.train[idx]
            train_targets = self.train_targets[idx]
        else:
            train = self.train
            train_targets = self.train_targets

        cfg = T.ConfigProto(intra_op_parallelism_threads=1,
                            inter_op_parallelism_threads=1)
        session = T.Session(config=cfg)
        K.set_session(session)

        pc = ParamFCNetClassification(config=x, n_feat=train.shape[1],
                                      n_classes=self.n_classes)
        history = pc.train(train, train_targets, self.valid, self.valid_targets,
                           n_epochs=self.n_epochs,
                           do_early_stopping=self.do_early_stopping)
        y = 1 - history.history["val_acc"][-1]

        if not np.isfinite(y):
            y = 1

        c = time.time() - start_time

        return {'function_value': y, "cost": c}

    @AbstractBenchmark._check_configuration
    def objective_function_test(self, x, **kwargs):
        start_time = time.time()

        rng = kwargs.get("rng", None)
        self.rng = rng_helper.get_rng(rng=rng,
                                      self_rng=self.rng)

        train = np.concatenate((self.train, self.valid))
        train_targets = np.concatenate((self.train_targets,
                                        self.valid_targets))

        pc = ParamFCNetClassification(config=x, n_feat=train.shape[1],
                                      n_classes=self.n_classes)
        history = pc.train(train, train_targets, self.test,
                           self.test_targets,
                           n_epochs=self.n_epochs,
                           do_early_stopping=self.do_early_stopping)
        y = 1 - history.history["val_acc"][-1]

        if not np.isfinite(y):
            y = 1

        c = time.time() - start_time

        return {'function_value': y, "cost": c}

    @staticmethod
    def get_configuration_space(max_num_layers=10):
        cs = ParamFCNetClassification.get_config_space(max_num_layers=max_num_layers)
        return cs

    @staticmethod
    def get_meta_information():
        info = dict()
        info['references'] = ["N.A.", ]
        info["cvfolds"] = 1
        info["wallclocklimit"] = 24*60*60
        info['num_function_evals'] = 100
        info['cutoff'] = 1800
        info['memorylimit'] = 1024 * 3
        return info


class ParamNetOnMnist(ParamNetBenchmark):

    def get_data(self):
        dm = MNISTData()
        return dm.load()

    @staticmethod
    def get_meta_information():
        d = ParamNetBenchmark.get_meta_information()
        d["references"].append("@article{lecun-ieee98,"
                               "title={Gradient-based learning applied to document recognition},"
                               "author={Y. LeCun and L. Bottou and Y. Bengio and P. Haffner},"
                               "journal={Proceedings of the IEEE},"
                               "pages={2278--2324},"
                               "year={1998},"
                               "publisher={IEEE}"
                               )
        return d


class ParamNetOnVehicle(ParamNetBenchmark):

    def get_data(self):
        dm = OpenMLHoldoutDataManager(openml_task_id=75191)
        X_train, y_train, X_valid, y_valid, X_test, y_test = dm.load()

        # Make sparse matrices dense
        X_train = X_train.toarray()
        X_valid = X_valid.toarray()
        X_test = X_test.toarray()

        # Zero mean / unit std normalization
        X_train, mean, std = zero_mean_unit_var_normalization(X_train)
        X_valid, _, _ = zero_mean_unit_var_normalization(X_valid, mean, std)
        X_test, _, _ = zero_mean_unit_var_normalization(X_test, mean, std)

        return X_train, y_train, X_valid, y_valid, X_test, y_test


class ParamNetOnCovertype(ParamNetBenchmark):

    def get_data(self):
        dm = OpenMLHoldoutDataManager(openml_task_id=2118)

        X_train, y_train, X_valid, y_valid, X_test, y_test = dm.load()

        # Zero mean / unit std normalization
        X_train, mean, std = zero_mean_unit_var_normalization(X_train)
        X_valid, _, _ = zero_mean_unit_var_normalization(X_valid, mean, std)
        X_test, _, _ = zero_mean_unit_var_normalization(X_test, mean, std)

        return X_train, y_train, X_valid, y_valid, X_test, y_test


class ParamNetOnLetter(ParamNetBenchmark):

    def get_data(self):
        dm = OpenMLHoldoutDataManager(openml_task_id=236)

        X_train, y_train, X_valid, y_valid, X_test, y_test = dm.load()

        # Zero mean / unit std normalization
        X_train, mean, std = zero_mean_unit_var_normalization(X_train)
        X_valid, _, _ = zero_mean_unit_var_normalization(X_valid, mean, std)
        X_test, _, _ = zero_mean_unit_var_normalization(X_test, mean, std)

        return X_train, y_train, X_valid, y_valid, X_test, y_test


class ParamNetOnOpenML100(ParamNetBenchmark):

    """Abstract Class to run ParamNet on OpenML100 datasets

    Attributes:
    -----------
    task_id : int
        Task id used to retrieve dataset from openml.org
    """

    def __init__(self, task_id, n_epochs=100, do_early_stopping=False,
                 rng=None):
        self.task_id = task_id
        super(ParamNetOnOpenML100, self).__init__()

    def get_data(self):
        """Gets data from OpenMl (downloads if necessary)

        Returns
        -------
        X_train, y_train, X_valid, y_valid, X_test, y_test
        """
        dm = OpenMLHoldoutDataManager(openml_task_id=self.task_id)

        X_train, y_train, X_valid, y_valid, X_test, y_test = dm.load()

        # Zero mean / unit std normalization
        X_train, mean, std = zero_mean_unit_var_normalization(X_train)
        X_valid, _, _ = zero_mean_unit_var_normalization(X_valid, mean, std)
        X_test, _, _ = zero_mean_unit_var_normalization(X_test, mean, std)

        return X_train, y_train, X_valid, y_valid, X_test, y_test

all_tasks = openml.tasks.list_tasks(task_type_id=1, tag='study_14')
for task_id in all_tasks:
    benchmark_string = """class ParamNet_OpenML100_%d(ParamNetOnOpenML100):

     def __init__(self, n_epochs=100, do_early_stopping=False, rng=None):
         super().__init__(%d, rng=rng) """ % (task_id, task_id)

    exec(benchmark_string)
