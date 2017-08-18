import time
import numpy as np

from param_net import ParamFCNetClassification
from param_net.util import zero_mean_unit_var_normalization
from sklearn.preprocessing import Imputer

from hpolib.util import rng_helper
from hpolib.util.data_manager import MNISTData
from hpolib.util.openml_data_manager import OpenMLHoldoutDataManager, \
    OpenMLCrossvalidationDataManager
from hpolib.abstract_benchmark import AbstractBenchmark

#import openml

from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

import keras.backend as K
import tensorflow as T


class ParamNetBenchmark(AbstractBenchmark):

    def __init__(self, n_epochs=100, early_stopping=False, rng=None):
        super(ParamNetBenchmark, self).__init__(rng=rng)

        self.train, self.train_targets, self.valid, self.valid_targets, \
            self.test, self.test_targets = self.get_data()
        self.n_epochs = n_epochs
        self.early_stopping = early_stopping

        # Use 10 time the number of classes as lower bound for the dataset
        # fraction
        self.n_classes = np.unique(self.train_targets).shape[0]
        self.s_min = float(10 * self.n_classes) / self.train.shape[0]

    def get_data(self):
        raise NotImplementedError()

    @AbstractBenchmark._check_configuration
    def objective_function(self, x, dataset_fraction=1, **kwargs):
        start_time = time.time()

        time_limit_s = kwargs.get("cutoff", None)
        rng = kwargs.get("rng", None)
        self.rng = rng_helper.get_rng(rng=rng, self_rng=self.rng)

        fold = int(float(kwargs.get("fold", 0)))
        folds = kwargs.get('folds', 1)

        if fold == folds:
            # Test fold, run function_test
            return self.objective_function_test(x, rng=rng)
        if fold < 0 or fold > folds:
            raise ValueError("%s is not a valid fold" % fold)

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

        y = self._train(x, train_X=train, train_y=train_targets,
                        valid_X=self.valid, valid_y=self.valid_targets,
                        time_limit_s=time_limit_s)

        c = time.time() - start_time

        return {'function_value': y, "cost": c}

    @AbstractBenchmark._check_configuration
    def objective_function_test(self, x, **kwargs):
        start_time = time.time()

        time_limit_s = kwargs.get("cutoff", None)
        rng = kwargs.get("rng", None)
        self.rng = rng_helper.get_rng(rng=rng,
                                      self_rng=self.rng)

        train = np.concatenate((self.train, self.valid))
        train_targets = np.concatenate((self.train_targets,
                                        self.valid_targets))

        y = self._train(x, train_X=train, train_y=train_targets,
                        valid_X=self.test, valid_y=self.test_targets,
                        time_limit_s=time_limit_s)

        c = time.time() - start_time

        return {'function_value': y, "cost": c}

    def _train(self, x, train_X, train_y, valid_X, valid_y, time_limit_s=None):
        np.random.seed(self.rng.randint(1))
        cfg = T.ConfigProto(intra_op_parallelism_threads=1,
                            inter_op_parallelism_threads=1)
        session = T.Session(config=cfg)
        print("Training on: ", train_y.shape, train_X.shape)
        print("Validate on: ", valid_y.shape, valid_X.shape)

        K.set_session(session)
        print("Use a timelimit of %s" % str(time_limit_s))
        pc = ParamFCNetClassification(config=x, n_feat=train_X.shape[1],
                                      n_classes=self.n_classes,
                                      max_num_epochs=self.n_epochs,
                                      metrics=['accuracy'], verbose=1,
                                      early_stopping=self.early_stopping)

        history = pc.train(train_X, train_y, valid_X, valid_y,
                           shuffle=True, n_epochs=self.n_epochs,
                           time_limit_s=time_limit_s)
        print(history)
        y = 1 - history["val_acc"][-1]

        if not np.isfinite(y):
            y = 1
        return y

    @staticmethod
    def get_configuration_space(max_num_layers=10):
        cs = ParamFCNetClassification.get_config_space(max_num_layers=max_num_layers,
                                                       batch_normalization=["True", "False"])
        return cs

    @staticmethod
    def get_meta_information():
        info = dict()
        info['references'] = ["N.A.", ]
        info["cvfolds"] = 1
        info["wallclocklimit"] = 24*60*60*2
        info['num_function_evals'] = 1000
        info['cutoff'] = 1800
        info['memorylimit'] = 1024 * 3
        info['compatible_with_commit'] = "6a887d36e80227b1d22821bd467fd2ef58d67933"
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
        dm = OpenMLHoldoutDataManager(openml_task_id=75191, rng=self.rng)
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
        dm = OpenMLHoldoutDataManager(openml_task_id=2118, rng=self.rng)

        X_train, y_train, X_valid, y_valid, X_test, y_test = dm.load()

        # Zero mean / unit std normalization
        X_train, mean, std = zero_mean_unit_var_normalization(X_train)
        X_valid, _, _ = zero_mean_unit_var_normalization(X_valid, mean, std)
        X_test, _, _ = zero_mean_unit_var_normalization(X_test, mean, std)

        return X_train, y_train, X_valid, y_valid, X_test, y_test


class ParamNetOnLetter(ParamNetBenchmark):

    def get_data(self):
        dm = OpenMLHoldoutDataManager(openml_task_id=236, rng=self.rng)

        X_train, y_train, X_valid, y_valid, X_test, y_test = dm.load()

        # Zero mean / unit std normalization
        X_train, mean, std = zero_mean_unit_var_normalization(X_train)
        X_valid, _, _ = zero_mean_unit_var_normalization(X_valid, mean, std)
        X_test, _, _ = zero_mean_unit_var_normalization(X_test, mean, std)

        return X_train, y_train, X_valid, y_valid, X_test, y_test


class ParamNetOnOpenML100_Holdout(ParamNetBenchmark):

    """Abstract Class to run ParamNet on OpenML100 datasets

    Attributes:
    -----------
    task_id : int
        Task id used to retrieve dataset from openml.org
    n_epochs : int
        Number of epochs to train this network
    early_stopping : bool
        Whether to do early stopping or not
    rng : int/np.random.RandomState
        Will be used to generate train/valid split
    """

    def __init__(self, task_id, n_epochs=100, early_stopping=False, rng=None):
        self.task_id = task_id
        super().__init__(n_epochs=n_epochs, early_stopping=early_stopping,
                         rng=rng)

    def get_data(self):
        """Abstract Class to run ParamNet on OpenML100 datasets using holdout

        Returns
        -------
        X_train, y_train, X_valid, y_valid, X_test, y_test
        """
        dm = OpenMLHoldoutDataManager(openml_task_id=self.task_id,
                                      rng=self.rng)

        X_train, y_train, X_valid, y_valid, X_test, y_test = dm.load()
        imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
        imp.fit(X_train)
        X_train = imp.transform(X_train)
        X_valid = imp.transform(X_valid)
        X_test = imp.transform(X_test)

        # Zero mean / unit std normalization
        _, mean, std = zero_mean_unit_var_normalization(X_train)
        std += 1e-8
        X_train, _, _ = zero_mean_unit_var_normalization(X_train, mean, std)
        X_valid, _, _ = zero_mean_unit_var_normalization(X_valid, mean, std)
        X_test, _, _ = zero_mean_unit_var_normalization(X_test, mean, std)

        return X_train, y_train, X_valid, y_valid, X_test, y_test


class ParamNetOnOpenML100_Crossvalidation(ParamNetBenchmark):

    """Abstract Class to run ParamNet on OpenML100 datasets using crossvalidation

    Attributes:
    -----------
    task_id : int
        Task id used to retrieve dataset from openml.org
    n_epochs : int
        Number of epochs to train this network
    early_stopping : bool
        Whether to do early stopping or not
    rng : int/np.random.RandomState
        Will be used to generate train/valid split
    """

    def __init__(self, task_id, n_epochs=100, early_stopping=False, rng=None):
        self.task_id = task_id
        self.folds = ParamNetOnOpenML100_Crossvalidation.\
                                              get_meta_information()["cvfolds"]
        super().__init__(n_epochs=n_epochs, early_stopping=early_stopping,
                         rng=rng)

    @staticmethod
    def get_meta_information():
        d = ParamNetBenchmark.get_meta_information()
        d["cvfolds"] = 10
        return d

    def get_data(self):
        """Gets data from OpenMl (downloads if necessary)
        *NOTE* also applies zero-mean unit-variance normalization

        Returns
        -------
        X_train, y_train, None, None, X_test, y_test
        """
        dm = OpenMLCrossvalidationDataManager(openml_task_id=self.task_id, rng=self.rng)
        X_train, y_train, X_test, y_test = dm.load()

        imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
        imp.fit(X_train)
        X_train = imp.transform(X_train)
        X_test = imp.transform(X_test)

        # Zero mean / unit std normalization
        _, mean, std = zero_mean_unit_var_normalization(X_train)
        std += 1e-8
        X_train, _, _ = zero_mean_unit_var_normalization(X_train, mean, std)
        X_test, _, _ = zero_mean_unit_var_normalization(X_test, mean, std)

        return X_train, y_train, None, None, X_test, y_test

    @AbstractBenchmark._check_configuration
    def objective_function(self, x, **kwargs):
        start_time = time.time()

        time_limit_s = kwargs.get("cutoff", None)
        rng = kwargs.get("rng", None)
        self.rng = rng_helper.get_rng(rng=rng, self_rng=self.rng)

        fold = int(float(kwargs.get("fold", 0)))
        folds = kwargs.get('folds', self.folds)

        if fold == folds:
            # Test fold, run function_test
            return self.objective_function_test(x, rng=rng)

        if fold < 0 or fold > folds:
            raise ValueError("%s is not a valid fold" % fold)

        # Compute crossvalidation splits
        kf = StratifiedKFold(n_splits=folds, shuffle=True,
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

        y = self._train(x=x, train_X=train, train_y=train_targets,
                        valid_X=valid, valid_y=valid_targets,
                        time_limit_s=time_limit_s)
        c = time.time() - start_time

        return {'function_value': y, "cost": c}

    @AbstractBenchmark._check_configuration
    def objective_function_test(self, x, **kwargs):
        # Need to redefine this as we do not have validation data
        start_time = time.time()

        time_limit_s = kwargs.get("cutoff", None)
        rng = kwargs.get("rng", None)
        self.rng = rng_helper.get_rng(rng=rng,
                                      self_rng=self.rng)

        y = self._train(x, train_X=self.train, train_y=self.train_targets,
                        valid_X=self.test, valid_y=self.test_targets,
                        time_limit_s=time_limit_s)

        c = time.time() - start_time

        return {'function_value': y, "cost": c}

all_tasks = [258, 259, 261, 262, 266, 267, 271, 273, 275, 279, 283, 288, 2120,
             2121, 2125, 336, 75093, 75092, 75095, 75097, 75099, 75103, 75107,
             75106, 75109, 75108, 75112, 75129, 75128, 75135, 146574, 146575,
             146572, 146573, 146578, 146579, 146576, 146577, 75154, 146582,
             146583, 75156, 146580, 75159, 146581, 146586, 146587, 146584,
             146585, 146590, 146591, 146588, 146589, 75169, 146594, 146595,
             146592, 146593, 146598, 146599, 146596, 146597, 146602, 146603,
             146600, 146601, 75181, 146604, 146605, 75215, 75217, 75219, 75221,
             75225, 75227, 75231, 75230, 75232, 75235, 3043, 75236, 75239, 3047,
             232, 233, 236, 3053, 3054, 3055, 241, 242, 244, 245, 246, 248, 250,
             251, 252, 253, 254]
# all_tasks = openml.tasks.list_tasks(task_type_id=1, tag='study_14')
for task_id in all_tasks:
    benchmark_string = """class ParamNet_OpenML100_HO_%d(ParamNetOnOpenML100_Holdout):

     def __init__(self, n_epochs=100, early_stopping=False, rng=None):
         super().__init__(%d, n_epochs=n_epochs, early_stopping=early_stopping,
         rng=rng) """ % (task_id, task_id)

    exec(benchmark_string)

# all_tasks = openml.tasks.list_tasks(task_type_id=1, tag='study_14')
for task_id in all_tasks:
    benchmark_string = """class ParamNet_OpenML100_CV_%d(ParamNetOnOpenML100_Crossvalidation):

     def __init__(self, n_epochs=100, early_stopping=False, rng=None):
         super().__init__(%d, n_epochs=n_epochs, early_stopping=early_stopping,
         rng=rng) """ % (task_id, task_id)

    exec(benchmark_string)
