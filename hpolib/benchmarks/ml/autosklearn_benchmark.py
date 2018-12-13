import json
import logging
import os
import pickle
import tempfile

import lockfile

import autosklearn
import autosklearn.evaluation
import autosklearn.data.xy_data_manager
import autosklearn.util.backend
import autosklearn.constants
import autosklearn.util.pipeline
import autosklearn.metrics
import numpy as np

from hpolib.abstract_benchmark import AbstractBenchmark
from hpolib.util.dependencies import verify_packages
from hpolib.util.openml_data_manager import OpenMLCrossvalidationDataManager
from hpolib.util import rng_helper
import hpolib

__version__ = 0.2


class AutoSklearnBenchmark(AbstractBenchmark):
    """Base class for auto-sklearn benchmarks.

    auto-sklearn benchmarks implement Section 6 of the paper 'Efficient and
    Robust Automated Machine Learning' by Feurer et al., published in
    Proceedings of NIPS 2015.

    Parameters
    ----------

    task_id : int
        OpenML task ID.
    resampling_strategy : int
        Can be either ``'cv'`` or ``'partial-cv'``. ``'cv'`` is only
        implemented as a fallback for optimizers which cannot natively handle
        instances.
    """

    def __init__(self, task_id, resampling_strategy='partial-cv', rng=None):

        self._check_dependencies()
        self._get_data_manager(task_id)
        self._tmpdir = tempfile.mkdtemp()
        self._setup_backend()

        # Setup of the datamanager etc has to be done before call to super
        # which itselfs calls get_hyperparameter_configuration_space
        super().__init__(rng)

        if resampling_strategy in ['partial-cv', 'cv']:
            self.resampling_strategy = resampling_strategy
        else:
            raise ValueError(
                "A resampling strategy other than 'partial-cv' or 'cv'"
                "does not reproduce Section 6 of the Auto-sklearn paper."
            )
        self.metric = autosklearn.metrics.balanced_accuracy

    def __del__(self):
        for i in range(5):
            try:
                os.rmdir(self._tmpdir)
            except:
                pass

    def _setup_backend(self):
        tmp_folder = os.path.join(self._tmpdir, 'tmp')
        output_folder = os.path.join(self._tmpdir, 'out')
        self.backend = autosklearn.util.backend.create(
            temporary_directory=tmp_folder,
            output_directory=output_folder,
            delete_tmp_folder_after_terminate=True,
            delete_output_folder_after_terminate=True)
        self.backend.save_datamanager(self.data_manager)
        self.logger = logging.getLogger(self.__class__.__name__)

    def _get_data_manager(self, task_id):
        cache_dir = hpolib._config.data_dir
        autosklearn_cache_dir = os.path.join(cache_dir, 'auto-sklearn')
        try:
            os.makedirs(autosklearn_cache_dir)
        except:
            pass

        data_manager_cache_file = os.path.join(autosklearn_cache_dir,
                                               '%d.pkl' % task_id)
        if os.path.exists(data_manager_cache_file):
            with open(data_manager_cache_file, 'rb') as fh:
                self.data_manager = pickle.load(fh)
        else:
            dm = OpenMLCrossvalidationDataManager(task_id)
            X_train, y_train, X_test, y_test = dm.load()

            num_classes = len(np.unique(y_train))
            if num_classes == 2:
                task_type = autosklearn.constants.BINARY_CLASSIFICATION
            elif num_classes > 2:
                task_type = autosklearn.constants.MULTICLASS_CLASSIFICATION
            else:
                raise ValueError('This benchmark needs at least two classes.')

            variable_types = dm.variable_types
            name = dm.name

            # TODO in the future, the XYDataManager should have this as it's own
            # attribute
            data_manager = autosklearn.data.xy_data_manager.XYDataManager(
                X=X_train, y=y_train,
                X_test=X_test, y_test=y_test,
                task=task_type, feat_type=variable_types,
                dataset_name=name)

            with lockfile.LockFile(data_manager_cache_file):
                with open(data_manager_cache_file, 'wb') as fh:
                    pickle.dump(data_manager, fh)

            self.data_manager = data_manager

    def _check_dependencies(self):
        dependencies = ['numpy>=1.9.0',
                        'scipy>=0.14.1',
                        'scikit-learn>=0.19.0',
                        'pynisher==0.5.0',
                        'auto-sklearn==0.4.2']
        dependencies = '\n'.join(dependencies)
        verify_packages(dependencies)

    @staticmethod
    def get_meta_information():
        info = {'name': 'auto-sklearn',
                'references': ["""@incollection{NIPS2015_5872,
title = {Efficient and Robust Automated Machine Learning},
author = {Feurer, Matthias and Klein, Aaron and Eggensperger, Katharina and Springenberg, Jost and Blum, Manuel and Hutter, Frank},
booktitle = {Advances in Neural Information Processing Systems 28},
editor = {C. Cortes and N. D. Lawrence and D. D. Lee and M. Sugiyama and R. Garnett},
pages = {2962--2970},
year = {2015},
publisher = {Curran Associates, Inc.},
url = {http://papers.nips.cc/paper/5872-efficient-and-robust-automated-machine-learning.pdf}
}"""]}
        info["cvfolds"] = 10
        info["wallclocklimit"] = 24 * 60 * 60
        info['num_function_evals'] = np.inf
        info['cutoff'] = 1800
        info['memorylimit'] = 1024 * 3

        return info

    @AbstractBenchmark._check_configuration
    def objective_function(self, configuration, **kwargs):
        """This objective function contains a specific fall-back for SMAC
        which requires the optimization target to be the same for training
        and testing and only differ by the instance (fold). Therefore,
        if the fold passed equals the number of folds, the objective function
        will evaluate the configuration on the test set instead of the
        validation set."""

        folds = kwargs.get('folds', 10)
        cutoff = kwargs.get('cutoff', 1800)
        memory_limit = kwargs.get('memory_limit', 3072)
        subsample = kwargs.get('subsample', None)

        # (TODO) For now ignoring seed
        rng = kwargs.get("rng")
        self.rng = rng_helper.get_rng(rng=rng, self_rng=self.rng)

        if self.resampling_strategy == 'partial-cv':
            fold = kwargs['fold']
            instance = json.dumps({'fold': fold, 'subsample': subsample})
        else:
            fold = None
            instance = None

        if fold and fold == folds:
            # run validation with the same memory limit and cutoff
            return self.objective_function_test(configuration,
                                                cutoff=cutoff,
                                                rng=rng,
                                                memory_limit=memory_limit)

        else:
            include, _ = self._get_include_exclude_info()
            evaluator = autosklearn.evaluation.ExecuteTaFuncWithQueue(
                backend=self.backend,
                autosklearn_seed=1,
                resampling_strategy=self.resampling_strategy,
                folds=folds,
                logger=self.logger,
                memory_limit=memory_limit,
                metric=self.metric,
                include=include,
            )

            status, cost, runtime, additional_run_info = evaluator.run(
                config=configuration, cutoff=cutoff, instance=instance)

            return {
                'function_value': cost,
                'cost': runtime,
                'status': status,
                'additional_run_info': additional_run_info,
            }

    @AbstractBenchmark._check_configuration
    def objective_function_test(self, configuration, **kwargs):
        """To compensate that the configuration found during configuration
        are now trained on a larger dataset, the cutoff and the memory limit
        are both increased.
        """
        cutoff = kwargs.get('cutoff', 3600)
        memory_limit = kwargs.get('memory_limit', 6144)
        instance = json.dumps({})

        include, _ = self._get_include_exclude_info()
        test_evaluator = autosklearn.evaluation.ExecuteTaFuncWithQueue(
            backend=self.backend,
            autosklearn_seed=1,
            resampling_strategy='test',
            logger=self.logger,
            memory_limit=memory_limit,
            metric=self.metric,
            include=include)

        status, cost, runtime, additional_run_info = test_evaluator.run(
            config=configuration, cutoff=cutoff, instance=instance)

        return {'function_value': cost, 'cost': runtime,
                'status': status, 'additional_run_info': additional_run_info}

    def _get_include_exclude_info(self):
        include = {'classifier': ['adaboost', 'bernoulli_nb', 'decision_tree',
                                  'extra_trees', 'gaussian_nb',
                                  'gradient_boosting', 'k_nearest_neighbors',
                                  'lda', 'liblinear_svc', 'libsvm_svc',
                                  'multinomial_nb', 'passive_aggressive', 'qda',
                                  'random_forest', 'sgd']}
        exclude = {}
        return include, exclude


class MulticlassClassificationBenchmark(AutoSklearnBenchmark):

    def get_configuration_space(self):
        include, exclude = self._get_include_exclude_info()
        task = autosklearn.constants.MULTICLASS_CLASSIFICATION
        sparse = self.data_manager.info['is_sparse']
        cs = autosklearn.util.pipeline.get_configuration_space(
            info={'task': task, 'is_sparse': sparse},
            include_estimators=include.get('classifier'),
            include_preprocessors=include.get('preprocessor'),
            exclude_estimators=exclude.get('classifier'),
            exclude_preprocessors=exclude.get('preprocessor'))
        return cs


class Sick(MulticlassClassificationBenchmark):
    """A.K.A. OpenML ID 38."""
    def __init__(self, resampling_strategy='partial-cv', rng=None):
        super().__init__(3043, resampling_strategy=resampling_strategy, rng=rng)


class Splice(MulticlassClassificationBenchmark):
    """A.K.A. OpenML ID 46."""

    def __init__(self, resampling_strategy='partial-cv', rng=None):
        super().__init__(275, resampling_strategy=resampling_strategy, rng=rng)


class Adult(MulticlassClassificationBenchmark):
    """A.K.A. OpenML ID 179."""

    def __init__(self, resampling_strategy='partial-cv', rng=None):
        super().__init__(2117, resampling_strategy=resampling_strategy, rng=rng)

    @staticmethod
    def get_meta_information():
        d = AutoSklearnBenchmark.get_meta_information()

        d["references"].append("""@inproceedings{Kohavi_kdd96,
title={Scaling Up the Accuracy of Naive-Bayes Classifiers: A Decision-Tree Hybrid.},
author={Kohavi, Ron},
booktitle={Proceedings of the Second International Conference on Knowledge Discovery and Data Mining},
volume={96},
pages={202--207},
year={1996}
}""")
        return d


class KROPT(MulticlassClassificationBenchmark):
    """A.K.A. OpenML ID 184."""

    def __init__(self, resampling_strategy='partial-cv', rng=None):
        super().__init__(2122, resampling_strategy=resampling_strategy, rng=rng)


class MNIST(MulticlassClassificationBenchmark):
    """A.K.A. OpenML ID 554."""

    def __init__(self, resampling_strategy='partial-cv', rng=None):
        super().__init__(
            75098,
            resampling_strategy=resampling_strategy,
            rng=rng,
        )


class Quake(MulticlassClassificationBenchmark):
    """A.K.A. OpenML ID 772."""

    def __init__(self, resampling_strategy='partial-cv', rng=None):
        super().__init__(
            75157,
            resampling_strategy=resampling_strategy,
            rng=rng,
        )


class fri_c1_1000_25(MulticlassClassificationBenchmark):
    """A.K.A. OpenML ID 917."""

    def __init__(self, resampling_strategy='partial-cv', rng=None):
        super().__init__(
            75209,
            resampling_strategy=resampling_strategy,
            rng=rng,
        )


class PC4(MulticlassClassificationBenchmark):
    """A.K.A. OpenML ID 1049."""

    def __init__(self, resampling_strategy='partial-cv', rng=None):
        super().__init__(
            75092,
            resampling_strategy=resampling_strategy,
            rng=rng,
        )


class KDDCup09_appetency(MulticlassClassificationBenchmark):
    """A.K.A. OpenML ID 1111."""

    def __init__(self, resampling_strategy='partial-cv', rng=None):
        super().__init__(
            75105,
            resampling_strategy=resampling_strategy,
            rng=rng,
        )


class MagicTelescope(MulticlassClassificationBenchmark):
    """A.K.A. OpenML ID 1120."""

    def __init__(self, resampling_strategy='partial-cv', rng=None):
        super().__init__(
            75112,
            resampling_strategy=resampling_strategy,
            rng=rng,
        )


class OVABreast(MulticlassClassificationBenchmark):
    """A.K.A. OpenML ID 1128."""

    def __init__(self, resampling_strategy='partial-cv', rng=None):
        super().__init__(
            75114,
            resampling_strategy=resampling_strategy,
            rng=rng,
        )


class Covertype(MulticlassClassificationBenchmark):
    """A.K.A. OpenML ID 293."""

    def __init__(self, resampling_strategy='partial-cv', rng=None):
        super().__init__(
            75164,
            resampling_strategy=resampling_strategy,
            rng=rng,
        )


class FBIS_WC(MulticlassClassificationBenchmark):
    """A.K.A. OpenML ID 389."""

    def __init__(self, resampling_strategy='partial-cv', rng=None):
        super().__init__(
            75197,
            resampling_strategy=resampling_strategy,
            rng=rng,
        )


all_tasks = [
    258, 259, 261, 262, 266, 267, 271, 273, 275, 279, 283, 288, 2120,
    2121, 2125, 336, 75093, 75092, 75095, 75097, 75099, 75103, 75107,
    75106, 75109, 75108, 75112, 75129, 75128, 75135, 146574, 146575,
    146572, 146573, 146578, 146579, 146576, 146577, 75154, 146582,
    146583, 75156, 146580, 75159, 146581, 146586, 146587, 146584,
    146585, 146590, 146591, 146588, 146589, 75169, 146594, 146595,
    146592, 146593, 146598, 146599, 146596, 146597, 146602, 146603,
    146600, 146601, 75181, 146604, 146605, 75215, 75217, 75219, 75221,
    75225, 75227, 75231, 75230, 75232, 75235, 3043, 75236, 75239, 3047,
    232, 233, 236, 3053, 3054, 3055, 241, 242, 244, 245, 246, 248, 250,
    251, 252, 253, 254,
]

for task_id in all_tasks:
    benchmark_string = """class OpenML100_%d(MulticlassClassificationBenchmark):

     def __init__(self, resampling_strategy='partial-cv', rng=None):
         super().__init__(%d, resampling_strategy=resampling_strategy, rng=rng)
""" % (task_id, task_id)

    exec(benchmark_string)
