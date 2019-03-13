import copy
import csv
import gzip
import logging
import os
import pickle
from urllib.request import urlretrieve

import lockfile

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
import sklearn.ensemble
import sklearn.pipeline
import sklearn.preprocessing

import hpolib
from hpolib.abstract_benchmark import AbstractBenchmark


__version__ = 0.1


class ExploringOpenML(AbstractBenchmark):
    """Surrogate benchmarks based on the data from Automatic Exploration of Machine Learning
    Benchmarks on OpenML by Kühn et al..

    This is a base class that should not be used directly. Instead, use one of the automatically
    constructed classes at the bottom of the file.

    Data is obtained from:
    https://figshare.com/articles/OpenML_R_Bot_Benchmark_Data_final_subset_/5882230
    """
    url = None

    def __init__(self, dataset_id, n_splits=10, n_iterations=30, rebuild=False, rng=None):
        """

        Parameters
        ----------
        dataset_id: int
            Dataset Id as given in Table 2.
        n_splits : int
            Number of cross-validation splits for optimizing the surrogate hyperparameters.
        n_iterations : int
            Number of iterations of random search to construct a surrogate model
        rebuild : bool
            Whether to construct a new surrogate model if there is already one stored to disk.
            This is important because changing the ``n_splits`` and the ``n_iterations``
            arguments do not trigger a rebuild of the surrogate.
        rng: int/None/RandomState
            set up rng
        """

        super().__init__(rng=rng)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.dataset_id = dataset_id
        self.classifier = self.__class__.__name__.split('_')[0]

        surrogate_dir = os.path.join(hpolib._config.data_dir, "ExploringOpenML", 'surrogates')
        if not os.path.exists(surrogate_dir):
            os.makedirs(surrogate_dir, exist_ok=True)
        surrogate_file_name = os.path.join(
            surrogate_dir,
            'surrogate_%s_%d.pkl.gz' % (self.classifier, self.dataset_id),
        )
        if rebuild or not os.path.exists(surrogate_file_name):
            self.construct_surrogate(dataset_id, n_splits, n_iterations)
            with lockfile.LockFile(surrogate_file_name):
                with gzip.open(surrogate_file_name, 'wb') as fh:
                    pickle.dump(
                        (self.regressor_loss, self.regressor_runtime, self.f_opt, self.f_max),
                        fh,
                    )
        else:
            with lockfile.LockFile(surrogate_file_name):
                with gzip.open(surrogate_file_name, 'rb') as fh:
                    (
                        self.regressor_loss,
                        self.regressor_runtime,
                        self.f_opt,
                        self.f_max,
                    ) = pickle.load(fh)

    def construct_surrogate(self, dataset_id, n_splits, n_iterations_rs):
        self.logger.info('Could not find surrogate pickle, constructing the surrogate.')

        save_to = os.path.join(hpolib._config.data_dir, "ExploringOpenML")
        if not os.path.isdir(save_to):
            os.makedirs(save_to)
        csv_path = os.path.join(save_to, self.classifier + '.csv')
        if not os.path.exists(csv_path):
            self.logger.info('Could not find surrogate data, downloading from %s', self.url)
            urlretrieve(self.url, csv_path)
            self.logger.info('Finished downloading surrogate data.')

        evaluations = []
        line_no = []

        self.logger.info('Starting to read in surrogate data.')
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
        configurations = []
        features = []
        targets = []
        runtimes = []
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
                config = ConfigSpace.util.deactivate_inactive_hyperparameters(
                    configuration_space=self.configuration_space,
                    configuration=config,
                )
            except ValueError as e:
                print(line_no[i], config, evaluation)
                raise e
            self.configuration_space.check_configuration(config)
            array = config.get_array()
            features.append(array)
            configurations.append(config)
            # HPOlib is about minimization!
            targets.append(1 - float(evaluation[target_features]))
            runtimes.append(float(evaluation['runtime']))

        features = np.array(features)
        targets = np.array(targets) + 1e-14
        runtimes = np.array(runtimes) + 1e-14
        features = self.impute(features)
        self.logger.info('Finished reading in surrogate data.')

        self.logger.info('Start building the surrogate, this can take a few minutes...')
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
        # This makes HPO deterministic
        cs.seed(1)
        highest_correlations_loss = -np.inf
        highest_correlations_loss_by_fold = np.array((n_splits,)) * -np.inf
        highest_correlations_runtime = -np.inf
        highest_correlations_runtime_by_fold = np.array((n_splits,)) * -np.inf
        best_config_loss = cs.get_default_configuration()
        best_config_runtime = cs.get_default_configuration()
        for n_iterations in range(n_iterations_rs):
            self.logger.debug('Random search iteration %d/%d.', n_iterations, n_iterations_rs)
            check_loss = True
            new_config_loss = cs.sample_configuration()
            new_config_runtime = copy.deepcopy(new_config_loss)
            regressor_loss = self.get_unfitted_regressor(
                new_config_loss, categorical_features, 25,
            )
            regressor_runtime = self.get_unfitted_regressor(
                new_config_runtime, categorical_features, 25,
            )

            rank_correlations_loss = np.ones((n_splits, )) * -np.NaN
            rank_correlations_runtime = np.ones((n_splits, )) * -np.NaN
            for n_fold, (train_idx, test_idx) in enumerate(
                    cv.split(features, targets)
            ):
                train_features = features[train_idx]
                train_targets_loss = targets[train_idx]
                train_targets_runtime = runtimes[train_idx]

                regressor_loss.fit(train_features, np.log(train_targets_loss))
                regressor_runtime.fit(train_features, np.log(train_targets_runtime))

                test_features = features[test_idx]

                y_hat_loss = np.exp(regressor_loss.predict(test_features))
                y_hat_runtime = np.exp(regressor_runtime.predict(test_features))

                test_targets_loss = targets[test_idx]
                spearman_rank_loss = scipy.stats.spearmanr(test_targets_loss, y_hat_loss)[0]
                rank_correlations_loss[n_fold] = spearman_rank_loss

                test_targets_runtime = runtimes[test_idx]
                spearman_rank_runtime = scipy.stats.spearmanr(
                    test_targets_runtime, y_hat_runtime,
                )[0]
                rank_correlations_runtime[n_fold] = spearman_rank_runtime

                if (
                    np.nanmean(highest_correlations_loss) * 0.99
                    > np.nanmean(rank_correlations_loss)
                ) and (
                    (
                        np.nanmean(highest_correlations_loss_by_fold[: n_splits + 1])
                        * (0.99 + n_fold * 0.001)
                    )
                    > np.nanmean(rank_correlations_loss[: n_splits + 1])
                ) and (
                    np.nanmean(highest_correlations_runtime) * 0.99
                    > np.nanmean(rank_correlations_runtime)
                ) and (
                    (
                        np.nanmean(highest_correlations_runtime_by_fold[: n_splits + 1])
                        * (0.99 + n_fold * 0.001)
                    )
                    > np.nanmean(rank_correlations_runtime[: n_splits + 1])
                ):
                    check_loss = False
                    break

            if (
                check_loss
                and np.mean(rank_correlations_loss) > highest_correlations_loss
            ):
                highest_correlations_loss = np.mean(rank_correlations_loss)
                highest_correlations_loss_by_fold = rank_correlations_loss
                best_config_loss = new_config_loss
            if (
                check_loss
                and np.mean(rank_correlations_runtime) > highest_correlations_runtime
            ):
                highest_correlations_runtime = np.mean(rank_correlations_runtime)
                highest_correlations_runtime_by_fold = rank_correlations_runtime
                best_config_runtime = new_config_runtime

        regressor_loss = self.get_unfitted_regressor(best_config_loss, categorical_features, 500)
        regressor_loss.fit(
            X=features,
            y=np.log(targets),
        )
        regressor_runtime = self.get_unfitted_regressor(
            best_config_runtime, categorical_features, 500,
        )
        regressor_runtime.fit(
            X=features,
            y=np.log(runtimes)
        )
        self.logger.info('Finished building the surrogate.')

        # Obtain the configuration for the best predictable value
        predictions = regressor_loss.predict(features)
        argmin = np.argmin(predictions)
        argmax = np.argmax(predictions)
        self.f_opt = configurations[argmin]
        self.f_max = configurations[argmax]
        self.regressor_loss = regressor_loss
        self.regressor_runtime = regressor_runtime

    def get_unfitted_regressor(self, config, categorical_features, n_trees):
        return sklearn.pipeline.Pipeline([
            (
                'ohe',
                sklearn.preprocessing.OneHotEncoder(
                    categorical_features=categorical_features,
                    sparse=False,
                ),
            ),
            ('poly', sklearn.preprocessing.PolynomialFeatures(
                degree=2,
                interaction_only=True,
                include_bias=False,
            )),
            ('estimator', sklearn.ensemble.RandomForestRegressor(
                n_estimators=n_trees,
                n_jobs=1,
                random_state=1,
                **config.get_dictionary()
            ))
        ])

    def impute(self, features):
        features = features.copy()
        for i, hp in enumerate(self.configuration_space.get_hyperparameters()):
            nan_rows = ~np.isfinite(features[:, i])
            features[nan_rows, i] = -1
        return features

    @AbstractBenchmark._check_configuration
    def objective_function(self, x, **kwargs):
        x = x.get_array().reshape((1, -1))
        x = self.impute(x)
        y = self.regressor_loss.predict(x)
        y = y[0]
        runtime = self.regressor_runtime.predict(x)
        runtime = runtime[0]
        # Untransform and round to the resolution of the data file.
        y = np.round(np.exp(y) - 1e-14, 6)
        runtime = np.round(np.exp(runtime) - 1e-14, 6)
        return {'function_value': y, 'cost': runtime}

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
    author = {Daniel Kühn and Philipp Probst and Janek Thomas and Bernd Bischl},
    year = {2018},
    }""", """@inproceedings{eggensperger_aaai2015a,
   author = {Katharina Eggensperger and Frank Hutter and Holger H. Hoos and Kevin Leyton-Brown},
   title = {Efficient Benchmarking of Hyperparameter Optimizers via Surrogates},
   booktitle = {Proceedings of the Twenty-Ninth AAAI Conference on Artificial Intelligence},
   conference = {AAAI Conference},
   year = {2015},
}

    """
            ]
        }

    def get_empirical_f_opt(self):
        """Return the empirical f_opt.

        Because ``get_meta_information`` is a static function it has no access to the actual
        function values predicted by the surrogate. This helper function gives access.

        Returns
        -------
        Configuration
        """
        return self.f_opt

    def get_empirical_f_max(self):
        """Return the empirical f_max.

        This is the configuration resulting in the worst predictive performance. Necessary to
        compute the average distance to the minimum metric typically used by Wistuba,
        Schilling and Schmidt-Thieme.

        Returns
        -------
        Configuration
        """
        return self.f_max


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
    3, 31, 37, 44, 50, 151, 312, 333, 334, 335, 1036, 1038, 1043, 1046, 1049,
    1050, 1063, 1067, 1068, 1120, 1461, 1462, 1464, 1467, 1471, 1479, 1480,
    1485, 1486, 1487, 1489, 1494, 1504, 1510, 1570, 4134, 4534,
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
         def __init__(self, n_splits=10, n_iterations=30, rng=None):
             super().__init__(dataset_id=%d, n_splits=n_splits, n_iterations=n_iterations, rebuild=False, rng=rng)
    """ % (model_class.__name__, dataset_id_, model_class.__name__, dataset_id_)

        exec(benchmark_string)


if __name__ == '__main__':
    # Call this script to construct all surrogates
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
