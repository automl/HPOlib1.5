import time

import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn import preprocessing, pipeline


import ConfigSpace as CS

from hpolib.abstract_benchmark import AbstractBenchmark
from hpolib.util.openml_data_manager import OpenMLHoldoutDataManager
import hpolib.util.rng_helper as rng_helper
from hpolib.util.data_manager import MNISTData


class XGBoost(AbstractBenchmark):

    def __init__(self, max_iterations=100, n_estimators=512, n_threads=1, rng=None):
        """
        XGBoost (https://github.com/dmlc/xgboost)
        Parameters
        ----------
        rng: str
            set up rng
        """
        super(XGBoost, self).__init__(rng=rng)

        self.X_trn, self.y_trn, self.X_val, self.y_val, self.X_tst, self.y_tst = self.get_data()
        self.max_iterations = max_iterations
        self.n_estimators = n_estimators
        self.n_threads = n_threads

    def get_data(self):
        raise NotImplementedError("Do not use this class as this is only a skeleton for further implementations.")

    @AbstractBenchmark._check_configuration
    @AbstractBenchmark._configuration_as_array
    def objective_function(self, x, **kwargs):
        learning_rate = 10**x[0]
        subsample = x[1]
        colsample_bytree = x[2]
        colsample_bylevel = x[3]

        rng = kwargs.get("rng", None)
        self.rng = rng_helper.get_rng(rng=rng, self_rng=self.rng)
        start = time.time()
        model = self._get_pipeline(learning_rate=learning_rate,
                                   subsample=subsample,
                                   colsample_bytree=colsample_bytree,
                                   colsample_bylevel=colsample_bylevel)
        model.fit(X=self.X_trn, y=self.y_trn)
        val_loss = self._test_xgb(model=model, X=self.X_val, y=self.y_val)
        cost = time.time() - start
        return {'function_value': val_loss, "cost": cost}

    @AbstractBenchmark._check_configuration
    @AbstractBenchmark._configuration_as_array
    def objective_function_test(self, x, **kwargs):
        learning_rate = 10**x[0]
        subsample = x[1]
        colsample_bytree = x[2]
        colsample_bylevel = x[3]

        rng = kwargs.get("rng", None)
        self.rng = rng_helper.get_rng(rng=rng, self_rng=self.rng)
        start = time.time()
        model = self._get_pipeline(learning_rate=learning_rate,
                                   subsample=subsample,
                                   colsample_bytree=colsample_bytree,
                                   colsample_bylevel=colsample_bylevel)
        model.fit(X=np.concatenate((self.X_trn, self.X_val)),
                  y=np.concatenate((self.y_trn, self.y_val)),)
        tst_loss = self._test_xgb(model=model, X=self.X_tst, y=self.y_tst)
        cost = time.time() - start
        return {'function_value': tst_loss, "cost": cost}

    @staticmethod
    def get_configuration_space():
        cs = CS.ConfigurationSpace(seed=np.random.randint(1, 100000))
        cs.generate_all_continuous_from_bounds(XGBoost.get_meta_information()['bounds'])
        return cs

    @staticmethod
    def get_meta_information():
        return {'name': 'XGBoost',
                'bounds': [[-3.0, 0.0], # learning rate
                           [0.01, 1.0], # subsample
                           [0.10, 1.0], # colsample bytree
                           [0.10, 1.0], # colsample bylevel
                ],
                'num_function_evals': 50,
                'requires': ["xgboost", ],
                'references': ["None"]
                }

    def _get_pipeline(self, learning_rate, subsample, colsample_bytree, colsample_bylevel):
        clf = pipeline.Pipeline([# No normalizing as it should not make a difference
                                 #('preproc', preprocessing.StandardScaler()),
                                 ('xgb', xgb.XGBClassifier(learning_rate=learning_rate, n_estimators=self.n_estimators,
                                                           objective='binary:logistic', n_jobs=self.n_threads,
                                                           subsample=subsample, colsample_bytree=colsample_bytree,
                                                           colsample_bylevel=colsample_bylevel,
                                                           random_state=self.rng.randint(1, 100000)))])
        return clf

    def _test_xgb(self, X, y, model):
        y_pred = model.predict(X)
        acc = accuracy_score(y_pred=y_pred, y_true=y)
        return 1-acc


class XGBoostOnHiggs(XGBoost):

    def get_data(self):
        dm = OpenMLHoldoutDataManager(openml_task_id=75101, rng=self.rng)
        X_train, y_train, X_val, y_val, X_test, y_test = dm.load()
        return X_train, y_train, X_val, y_val, X_test, y_test

    def _get_pipeline(self, learning_rate, subsample, colsample_bytree, colsample_bylevel):
        clf = pipeline.Pipeline([('preproc1', preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)),
                                 # No normalizing as it should not make a difference
                                 #('preproc2', preprocessing.StandardScaler()),
                                 ('xgb', xgb.XGBClassifier(learning_rate=learning_rate, n_estimators=self.n_estimators,
                                  objective='binary:logistic', n_jobs=self.n_threads,
                                  subsample=subsample, colsample_bytree=colsample_bytree,
                                  colsample_bylevel=colsample_bylevel, random_state=self.rng.randint(1, 100000)))])
        return clf


class XGBoostOnMnist(XGBoost):

    def get_data(self):
        dm = MNISTData()
        return dm.load()
