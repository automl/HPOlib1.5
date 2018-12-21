import time
import csv
import ConfigSpace
import numpy as np
import random
import xgboost as xgb

from hpolib.abstract_benchmark import AbstractBenchmark
import hpolib.util.rng_helper as rng_helper
from sklearn.metrics import mean_squared_error


class XGBoostRegressionUCI(AbstractBenchmark):
    """

    """
    def __init__(self, data_location, rng=None):
        """

        Parameters
        ----------
        rng: int/None/RandomState
            set up rng
        """

        self.train, self.train_targets, self.valid, self.valid_targets, \
            self.test, self.test_targets = self.get_data(data_location)

        super(XGBoostRegressionUCI, self).__init__()

        self.rng = rng_helper.create_rng(rng)

    def get_data(self, data_location, delimiter=","):
        f = open(data_location)
        data = []
        reader = csv.reader(f, delimiter=delimiter)
        rownum = 0
        for row in reader:
            if rownum == 0:
                # cut off header
                header = row
            else:
                colnum = 0
                v = []
                for col in row:
                    v.append(float(col))
                    colnum += 1
                data.append(v)
            rownum += 1
        data = np.array(data)

        X = data[:, :-1]

        # we assume that the last column specifies the targets
        y = data[:, -1]

        idx = np.arange(X.shape[0])
        random.shuffle(idx)
        X = X[idx]
        y = y[idx]

        n_train = int(X.shape[0] * 0.6)
        n_valid = int(X.shape[0] * 0.2)

        X_train = X[:n_train]
        X_valid = X[n_train:(n_train + n_valid)]
        X_test = X[(n_train + n_valid):]

        y_train = y[:n_train]
        y_valid = y[n_train:(n_train + n_valid)]
        y_test = y[(n_train + n_valid):]

        return X_train, y_train, X_valid, y_valid, X_test, y_test

    def objective_function(self, configuration, **kwargs):

        start_time = time.time()

        xgb_model = xgb.XGBRegressor(learning_rate=configuration["learning_rate"],
                                     gamma=configuration["gamma"],
                                     reg_alpha=configuration["reg_alpha"],
                                     reg_lambda=configuration["reg_lambda"],
                                     n_estimators=configuration["n_estimators"],
                                     subsample=configuration["subsample"],
                                     max_depth=configuration["max_depth"],
                                     min_child_weight=configuration["min_child_weight"])

        xgb_model.fit(self.train, self.train_targets)

        predictions = xgb_model.predict(self.valid)

        mse = mean_squared_error(self.valid_targets, predictions)

        runtime = time.time() - start_time

        return {'function_value': mse, "cost": runtime}

    def objective_function_test(self, configuration, **kwargs):

        start_time = time.time()

        train = np.concatenate((self.train, self.valid))
        train_targets = np.concatenate((self.train_targets, self.valid_targets))

        xgb_model = xgb.XGBRegressor(learning_rate=configuration["learning_rate"],
                                     gamma=configuration["gamma"],
                                     reg_alpha=configuration["reg_alpha"],
                                     reg_lambda=configuration["reg_lambda"],
                                     n_estimators=configuration["n_estimators"],
                                     subsample=configuration["subsample"],
                                     max_depth=configuration["max_depth"],
                                     min_child_weight=configuration["min_child_weight"])

        xgb_model.fit(train, train_targets)

        predictions = xgb_model.predict(self.test)

        mse = mean_squared_error(self.test_targets, predictions)

        runtime = time.time() - start_time

        return {'function_value': mse, "cost": runtime}

    @staticmethod
    def get_configuration_space():

        cs = ConfigSpace.ConfigurationSpace()

        cs.add_hyperparameter(ConfigSpace.UniformFloatHyperparameter(
            'learning_rate', lower=1e-6, upper=1e-1, log=True))

        cs.add_hyperparameter(ConfigSpace.UniformFloatHyperparameter(
            'gamma', lower=0, upper=2))

        cs.add_hyperparameter(ConfigSpace.UniformFloatHyperparameter(
            'reg_alpha', lower=1e-5, upper=1e3, log=True))

        cs.add_hyperparameter(ConfigSpace.UniformFloatHyperparameter(
            'reg_lambda', lower=1e-5, upper=1e3, log=True))

        cs.add_hyperparameter(ConfigSpace.UniformIntegerHyperparameter(
            'n_estimators', lower=10, upper=500, log=False))

        cs.add_hyperparameter(ConfigSpace.UniformFloatHyperparameter(
            'subsample', lower=1e-1, upper=1))

        cs.add_hyperparameter(ConfigSpace.UniformIntegerHyperparameter(
            'max_depth', lower=1, upper=15))

        cs.add_hyperparameter(ConfigSpace.UniformIntegerHyperparameter(
            'min_child_weight', lower=0, upper=20))

        return cs

    @staticmethod
    def get_meta_information():
        return {'name': 'XGBoost for regression on UCI datasets',
                'references': []
                }
