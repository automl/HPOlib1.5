import csv
import os
from urllib.request import urlretrieve

from ConfigSpace import (
    ConfigurationSpace,
    Configuration,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    CategoricalHyperparameter,
    EqualsCondition,
)
import ConfigSpace.util
import lightgbm
import numpy as np
import sklearn.metrics
import sklearn.model_selection

import hpolib
from hpolib.abstract_benchmark import AbstractBenchmark
from hpolib.util import rng_helper


class ExploringOpenML(AbstractBenchmark):
    """https://figshare.com/articles/OpenML_R_Bot_Benchmark_Data_final_subset_/5882230
    """

    def __init__(self, dataset_id, rng=None):
        """

        Parameters
        ----------
        dataset_id: int
            Dataset Id as given in Table 2.
        rng: int/None/RandomState
            set up rng
        """

        super().__init__(rng=rng)
        self.dataset_id = dataset_id
        self.classifier = self.__class__.__name__.split('_')[0]

        self.save_to = os.path.join(hpolib._config.data_dir,
                                    "ExploringOpenML")
        if not os.path.isdir(self.save_to):
            os.makedirs(self.save_to)
        csv_path = os.path.join(self.save_to, self.classifier + '.csv')
        if not os.path.exists(csv_path):
            urlretrieve(self.url, csv_path)

        evaluations = []
        line_no = []

        with open(csv_path) as fh:
            csv_reader = csv.DictReader(fh)
            for i, line in enumerate(csv_reader):
                if int(line['data_id']) != dataset_id:
                    continue
                evaluations.append(line)
                line_no.append(i)

        hyperparameters_names = [
            hp.name for hp in self.configuration_space.get_hyperparameters()
        ]
        categorical_features = [
            i for i in range(len(self.configuration_space.get_hyperparameters()))
            if isinstance(
                self.configuration_space.get_hyperparameters()[i],
                CategoricalHyperparameter,
            )
        ]
        target_features = 'auc'

        features = []
        targets = []
        for i, evaluation in enumerate(evaluations):
            number_of_features = float(evaluation['NumberOfFeatures'])
            number_of_datapoints = float(evaluation['NumberOfInstances'])
            config = {
                key: value
                for key, value
                in evaluation.items()
                if key in hyperparameters_names and value != 'NA'
            }
            # Do some specific transformations
            if self.classifier == 'Ranger':
                config['mtry'] = float(config['mtry']) / number_of_features
                config['min.node.size'] = (
                    np.log(float(config['min.node.size']))
                    / np.log(number_of_datapoints)
                )
                if config['min.node.size'] > 1.0:
                    # MF: according to Philipp it is unclear why this is in
                    # the data
                    continue
                if config['mtry'] > 1.0:
                    # MF: according to Philipp it is unclear why this is in
                    # the data
                    continue
            elif self.classifier == 'XGBoost':
                if 'eta' not in config:
                    # MF: according to Philipp, the algorithm was run in the
                    # default and the OpenML R package did not upload the
                    # default in one of the earliest versions
                    continue
                elif 'colsample_bytree' not in config:
                    # MF: according to Philipp, the algorithm was run in the
                    # default and the OpenML R package did not upload the
                    # default in one of the earliest versions
                    continue
                elif 'colsample_bylevel' not in config:
                    # MF: according to Philipp, the algorithm was run in the
                    # default and the OpenML R package did not upload the
                    # default in one of the earliest versions
                    continue
            config = ConfigSpace.util.fix_types(
                configuration=config,
                configuration_space=self.configuration_space,
            )
            try:
                config = Configuration(
                    configuration_space=self.configuration_space,
                    values=config,
                )
            except ValueError as e:
                print(line_no[i], config, evaluation)
                raise e
            array = config.get_array()
            features.append(array)
            targets.append(float(evaluation[target_features]))

        features = np.array(features)
        targets = np.array(targets)

        n_splits = 5
        cv = sklearn.model_selection.KFold(n_splits=n_splits, random_state=1)
        cs = ConfigurationSpace()
        max_depth = UniformIntegerHyperparameter('max_depth', 1, 10)
        learning_rate = UniformFloatHyperparameter(
            'learning_rate', 0.01, 0.2, log=True,
        )
        min_child_samples = UniformIntegerHyperparameter(
            'min_child_samples', 1, 50, log=True,
        )
        subsample = UniformFloatHyperparameter('subsample', 0.5, 1.0)
        colsample_bytree = UniformFloatHyperparameter(
            'colsample_bytree', 0.5, 1.0,
        )
        reg_alpha = UniformFloatHyperparameter(
            'reg_alpha', 1e-10, 1.0, log=True,
        )
        reg_lambda = UniformFloatHyperparameter(
            'reg_lambda', 1e-10, 1.0, log=True
        )
        cs.add_hyperparameters([
            max_depth, learning_rate, min_child_samples, subsample,
            colsample_bytree, reg_alpha, reg_lambda,])
        cs.seed(1)
        lowest_error = np.inf
        lowest_error_by_fold = np.ones((n_splits, )) * np.inf
        best_config = cs.get_default_configuration()
        n_iterations = 0
        while True:
            n_iterations += 1
            if lowest_error < 1 and n_iterations > 100:
                break
            check_loss = True
            new_config = cs.sample_configuration()
            regressor = lightgbm.LGBMRegressor(
                n_estimators=500,
                n_jobs=1,
                num_leaves=2**new_config['max_depth'],
                **new_config.get_dictionary()
            )
            cv_errors = np.ones((n_splits, )) * np.NaN
            for n_fold, (train_idx, test_idx) in enumerate(
                    cv.split(features, targets)
            ):
                train_features = features[train_idx]
                train_targets = targets[train_idx]

                regressor.fit(train_features, train_targets)

                test_features = features[test_idx]

                y_hat = regressor.predict(test_features)

                test_targets = targets[test_idx]
                error = np.sqrt(sklearn.metrics.mean_squared_error(test_targets, y_hat))
                cv_errors[n_fold] = error

                # Aggressive and simple pruning of folds
                if (
                    np.nanmean(lowest_error_by_fold) * 1.2
                    < np.nanmean(cv_errors)
                ) and (
                    np.nanmean(lowest_error_by_fold[: n_splits + 1]) * 1.2
                    < np.nanmean(cv_errors[: n_splits + 1])
                ):
                    check_loss = False
                    break
                # Reject large error immediately
                if cv_errors[n_fold] > 100:
                    check_loss = False
                    break

            if check_loss and np.mean(cv_errors) < lowest_error:
                lowest_error = np.mean(cv_errors)
                lowest_error_by_fold = cv_errors
                best_config = new_config

        regressor = lightgbm.LGBMRegressor(
            n_estimators=500,
            n_jobs=1,
            num_leaves=2**new_config['max_depth'],
            **best_config.get_dictionary()
        )
        regressor.fit(
            X=features,
            y=targets,
            categorical_feature=categorical_features,
        )
        self.regressor = regressor

    @AbstractBenchmark._check_configuration
    def objective_function(self, x, **kwargs):
        x = x.get_array().reshape((1, -1))
        y = self.regressor.predict(x)
        y = y[0]
        return {'function_value': y}

    @AbstractBenchmark._check_configuration
    def objective_function_test(self, x, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def get_meta_information():
        return {'num_function_evals': 50,
                'name': 'Exploring_OpenML',
                'f_opt': 0}


class GLMNET(ExploringOpenML):
    url = 'https://ndownloader.figshare.com/files/10462300'

    @staticmethod
    def get_configuration_space():
        cs = ConfigurationSpace()
        alpha = UniformFloatHyperparameter('alpha', lower=0, upper=1)
        lambda_ = UniformFloatHyperparameter(
            'lambda', lower=2**-10, upper=2**10, log=True,
        )
        cs.add_hyperparameters([alpha, lambda_])
        return cs


class RPART(ExploringOpenML):
    url = 'https://ndownloader.figshare.com/files/10462309'

    @staticmethod
    def get_configuration_space():
        cs = ConfigurationSpace()
        alpha = UniformFloatHyperparameter('cp', lower=0, upper=1)
        maxdepth = UniformIntegerHyperparameter(
            'maxdepth', lower=1, upper=30,
        )
        minbucket = UniformIntegerHyperparameter(
            'minbucket', lower=1, upper=60,
        )
        minsplit = UniformIntegerHyperparameter(
            'minsplit', lower=1, upper=60,
        )
        cs.add_hyperparameters([alpha, maxdepth, minbucket, minsplit])
        return cs


class KKNN(ExploringOpenML):
    url = 'https://ndownloader.figshare.com/files/10811312'

    @staticmethod
    def get_configuration_space():
        cs = ConfigurationSpace()
        k = UniformIntegerHyperparameter('k', lower=1, upper=30)
        cs.add_hyperparameters([k])
        return cs


class SVM(ExploringOpenML):
    url = 'https://ndownloader.figshare.com/files/10462312'

    @staticmethod
    def get_configuration_space():
        cs = ConfigurationSpace()
        kernel = CategoricalHyperparameter(
            'kernel', choices=['linear', 'polynomial', 'radial'],
        )
        cost = UniformFloatHyperparameter('cost', 2**-10, 2**10, log=True)
        gamma = UniformFloatHyperparameter('gamma', 2**-10, 2**10, log=True)
        degree = UniformIntegerHyperparameter('degree', 2, 5)
        cs.add_hyperparameters([kernel, cost, gamma, degree])
        gamma_condition = EqualsCondition(gamma, kernel, 'radial')
        degree_condition = EqualsCondition(degree, kernel, 'polynomial')
        cs.add_conditions([gamma_condition, degree_condition])
        return cs

class Ranger(ExploringOpenML):
    url = 'https://ndownloader.figshare.com/files/10462306'

    @staticmethod
    def get_configuration_space():
        cs = ConfigurationSpace()
        num_trees = UniformIntegerHyperparameter(
            'num.trees', lower=1, upper=2000,
        )
        replace = CategoricalHyperparameter(
            'replace', choices=['FALSE', 'TRUE'],
        )
        sample_fraction = UniformFloatHyperparameter(
            'sample.fraction', lower=0, upper=1,
        )
        mtry = UniformFloatHyperparameter(
            'mtry', lower=0, upper=1,
        )
        respect_unordered_factors = CategoricalHyperparameter(
            'respect.unordered.factors', choices=['FALSE', 'TRUE'],
        )
        min_node_size = UniformFloatHyperparameter(
            'min.node.size', lower=0, upper=1,
        )
        cs.add_hyperparameters([
            num_trees, replace, sample_fraction, mtry,
            respect_unordered_factors, min_node_size,
        ])
        return cs


class XGBoost(ExploringOpenML):
    url = 'https://ndownloader.figshare.com/files/10462315'

    @staticmethod
    def get_configuration_space():
        cs = ConfigurationSpace()
        nrounds = UniformIntegerHyperparameter(
            'nrounds', lower=1, upper=5000,
        )
        eta = UniformFloatHyperparameter(
            'eta', lower=2**-10, upper=2**0, log=True,
        )
        subsample = UniformFloatHyperparameter(
            'subsample', lower=0, upper=1,
        )
        booster = CategoricalHyperparameter(
            'booster', choices=['gblinear', 'gbtree'],
        )
        max_depth = UniformIntegerHyperparameter(
            'max_depth', lower=1, upper=15,
        )
        min_child_weight = UniformFloatHyperparameter(
            'min_child_weight', lower=2**0, upper=2**7, log=True,
        )
        colsample_bytree = UniformFloatHyperparameter(
            'colsample_bytree', lower=0, upper=1,
        )
        colsample_bylevel = UniformFloatHyperparameter(
            'colsample_bylevel', lower=0, upper=1,
        )
        lambda_ = UniformFloatHyperparameter(
            'lambda', lower=2**-10, upper=2**10, log=True,
        )
        alpha = UniformFloatHyperparameter(
            'alpha', lower=2**-10, upper=2**10, log=True,
        )
        cs.add_hyperparameters([
            nrounds, eta, subsample, booster, max_depth, min_child_weight,
            colsample_bytree, colsample_bylevel, lambda_, alpha,
        ])
        colsample_bylevel_condition = EqualsCondition(
            colsample_bylevel, booster, 'gbtree',
        )
        colsample_bytree_condition = EqualsCondition(
            colsample_bytree, booster, 'gbtree',
        )
        max_depth_condition = EqualsCondition(
            max_depth, booster, 'gbtree',
        )
        min_child_weight_condition = EqualsCondition(
            min_child_weight, booster, 'gbtree',
        )
        cs.add_conditions([
            colsample_bylevel_condition, colsample_bytree_condition,
            max_depth_condition, min_child_weight_condition,
        ])
        return cs


all_datasets = [
    3, #31, 37, 44, 50, 151, 312, 333, 334, 335, 1036, 1038, 1043, 1046, 1049,
    #1050, 1063, 1067, 1068, 1120, 1461, 1462, 1464, 1467, 1471, 1479, 1480,
    #1485, 1486, 1487, 1489, 1494, 1504, 1510, 1570, 4134, 4534,
]
all_model_classes = [
    GLMNET,
    RPART,
    KKNN,
    SVM,
    Ranger,
    XGBoost,
]

for model_class in all_model_classes:
    for dataset_id in all_datasets:
        benchmark_string = """class %s_%d(%s):
         def __init__(self, rng=None):
             super().__init__(dataset_id=%d, rng=rng)
    """ % (model_class.__name__, dataset_id, model_class.__name__, dataset_id)

        exec(benchmark_string)


if __name__ == '__main__':
    for model_class in all_model_classes:
        print(model_class)
        for dataset_id in all_datasets:
            print(dataset_id)
            exec('rval = %s_%d()' % (model_class.__name__, dataset_id))
            print(rval)
            cs = rval.get_configuration_space()
            for i in range(10):
                config = cs.sample_configuration()
                print(config, rval.objective_function(config))
