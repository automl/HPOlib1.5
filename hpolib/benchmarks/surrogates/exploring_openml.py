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
import numpy as np
import scipy.stats
import sklearn.metrics
import sklearn.model_selection
import sklearn.compose
import sklearn.ensemble
import sklearn.pipeline
import sklearn.preprocessing

import hpolib
from hpolib.abstract_benchmark import AbstractBenchmark


class ExploringOpenML(AbstractBenchmark):
    """https://figshare.com/articles/OpenML_R_Bot_Benchmark_Data_final_subset_/5882230
    """
    url = None

    def __init__(self, dataset_id, n_splits=10, rng=None):
        """

        Parameters
        ----------
        dataset_id: int
            Dataset Id as given in Table 2.
        n_splits : int
            Number of cross-validation splits for optimizing the surrogate hyperparameters.
        rng: int/None/RandomState
            set up rng
        """

        super().__init__(rng=rng)
        self.dataset_id = dataset_id
        self.classifier = self.__class__.__name__.split('_')[0]

        self.save_to = os.path.join(hpolib._config.data_dir, "ExploringOpenML")
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
            if float(evaluation[target_features]) > 1:
                raise ValueError(i, evaluation)
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
            # HPOlib is about minimization!
            targets.append(1 - float(evaluation[target_features]))

        features = np.array(features)
        targets = np.array(targets)

        self.impute_with_defaults(features)

        cv = sklearn.model_selection.KFold(n_splits=n_splits, random_state=1, shuffle=True)
        cs = ConfigurationSpace()
        min_samples_split = UniformIntegerHyperparameter(
            'min_samples_split', lower=2, upper=20, log=True,
        )
        min_samples_leaf = UniformIntegerHyperparameter('min_samples_leaf', 1, 20, log=True)
        max_features = UniformFloatHyperparameter('max_features', 0.5, 1.0)
        bootstrap = CategoricalHyperparameter('bootstrap', [True, False])

        cs.add_hyperparameters([
            min_samples_split,
            min_samples_leaf,
            max_features,
            bootstrap,
        ])
        cs.seed(1)

        highest_correlations = -np.inf
        highest_correlations_by_fold = np.array((n_splits, )) * -np.inf
        best_config = cs.get_default_configuration()
        n_iterations = 0

        while True:
            n_iterations += 1

            # Stopping condititions
            if n_iterations > 100:
                break
            if highest_correlations > 0.9 and n_iterations > 75:
                break
            elif highest_correlations > 0.95 and n_iterations > 50:
                break
            elif highest_correlations > 0.99 and n_iterations > 25:
                break
            elif highest_correlations > 0.999:
                break
            check_loss = True
            new_config = cs.sample_configuration()
            regressor = self.get_unfitted_regressor(new_config, categorical_features, features)

            rank_correlations = np.ones((n_splits, )) * -np.NaN
            for n_fold, (train_idx, test_idx) in enumerate(
                    cv.split(features, targets)
            ):
                print(n_iterations, n_fold)
                train_features = features[train_idx]
                train_targets = targets[train_idx]

                regressor.fit(train_features, np.log(train_targets))

                test_features = features[test_idx]

                y_hat = np.exp(regressor.predict(test_features))

                test_targets = targets[test_idx]
                spearman_rank = scipy.stats.spearmanr(test_targets, y_hat)[0]
                rank_correlations[n_fold] = spearman_rank

                if (
                    np.nanmean(highest_correlations) * 0.99
                    > np.nanmean(rank_correlations)
                ) and (
                    (
                        np.nanmean(highest_correlations_by_fold[: n_splits + 1])
                        * (0.99 + n_fold * 0.001)
                    )
                    > np.nanmean(rank_correlations[: n_splits + 1])
                ):
                    check_loss = False
                    break

            if check_loss and np.mean(rank_correlations) > highest_correlations:
                highest_correlations = np.mean(rank_correlations)
                highest_correlations_by_fold = rank_correlations
                best_config = new_config

        regressor = self.get_unfitted_regressor(best_config, categorical_features, features)
        regressor.fit(
            X=features,
            y=targets,
        )
        self.regressor = regressor

    def get_unfitted_regressor(self, config, categorical_features, features):
        return sklearn.pipeline.Pipeline([
            ('ct', sklearn.compose.ColumnTransformer([
                (
                    'numerical',
                    'passthrough',
                    [i for i in range(features.shape[1]) if i not in categorical_features],
                ),
                (
                    'categoricals',
                    sklearn.preprocessing.OneHotEncoder(categories='auto'),
                    categorical_features,
                ),
            ])),
            ('poly', sklearn.preprocessing.PolynomialFeatures(
                degree=2,
                interaction_only=True,
                include_bias=False,
            )),
            ('estimator', sklearn.ensemble.RandomForestRegressor(
                n_estimators=500,
                n_jobs=1,
                random_state=1,
                **config.get_dictionary()
            ))
        ])

    def impute_with_defaults(self, features):
        for i, hp in enumerate(self.configuration_space.get_hyperparameters()):
            nan_rows = ~np.isfinite(features[:, i])
            features[nan_rows, i] = hp.normalized_default_value

    @AbstractBenchmark._check_configuration
    def objective_function(self, x, **kwargs):
        x = x.get_array().reshape((1, -1))
        self.impute_with_defaults(x)
        y = self.regressor.predict(x)
        y = y[0]
        return {'function_value': y}

    @AbstractBenchmark._check_configuration
    def objective_function_test(self, x, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def get_meta_information():
        return {
            'num_function_evals': 50,
            'name': 'Exploring_OpenML',
            'f_opt': 0,
            'references': [
                """@article{kuhn_arxiv2018a,
    title = {Automatic {Exploration} of {Machine} {Learning} {Experiments} on {OpenML}},
    journal = {arXiv:1806.10961 [cs, stat]},
    author = {Kühn, Daniel and Probst, Philipp and Thomas, Janek and Bischl, Bernd},
    year = {2018},
    }""",
            ]
        }


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
    3,  # 31, 37, 44, 50, 151, 312, 333, 334, 335, 1036, 1038, 1043, 1046, 1049,
    # 1050, 1063, 1067, 1068, 1120, 1461, 1462, 1464, 1467, 1471, 1479, 1480,
    # 1485, 1486, 1487, 1489, 1494, 1504, 1510, 1570, 4134, 4534,
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
    for dataset_id_ in all_datasets:
        benchmark_string = """class %s_%d(%s):
         def __init__(self, rng=None):
             super().__init__(dataset_id=%d, rng=rng)
    """ % (model_class.__name__, dataset_id_, model_class.__name__, dataset_id_)

        exec(benchmark_string)


if __name__ == '__main__':
    for model_class in all_model_classes:
        print(model_class)
        for dataset_id_ in all_datasets:
            print(dataset_id_)
            exec('rval = %s_%d()' % (model_class.__name__, dataset_id_))
            print(rval)
            model_class_cs = rval.get_configuration_space()
            for _ in range(10):
                tmp_config = model_class_cs.sample_configuration()
                print(tmp_config, rval.objective_function(tmp_config))