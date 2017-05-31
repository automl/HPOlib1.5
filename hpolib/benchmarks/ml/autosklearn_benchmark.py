import json
import logging
import tempfile

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

__version__ = 0.1


class AutoSklearnBenchmark(AbstractBenchmark):
    """Base class for auto-sklearn benchmarks.

    auto-sklearn benchmarks implement Section 6 of the paper 'Efficient and
    Robust Automated Machine Learning' by Feurer et al., published in
    Proceedings of NIPS 2015.
    """

    def __init__(self, task_id, rng=None):

        self._check_dependencies()
        self._get_data_manager(task_id)
        self._setup_backend()
        self.metric = autosklearn.metrics.balanced_accuracy

        # Setup of the datamanager etc has to be done before call to super
        # which itselfs calls get_hyperparameter_configuration_space
        super().__init__(rng)

    def _setup_backend(self):
        tmp_folder = tempfile.mkdtemp()
        output_folder = tempfile.mkdtemp()
        self.backend = autosklearn.util.backend.create(
            temporary_directory=tmp_folder,
            output_directory=output_folder,
            delete_tmp_folder_after_terminate=True,
            delete_output_folder_after_terminate=True)
        self.backend.save_datamanager(self.data_manager)
        self.logger = logging.getLogger(self.__class__.__name__)

    def _get_data_manager(self, task_id):

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
            data_x=X_train, y=y_train, task=task_type, feat_type=variable_types,
            dataset_name=name)
        data_manager.data['X_test'] = X_test
        data_manager.data['Y_test'] = y_test

        self.data_manager = data_manager

    def _check_dependencies(self):
        dependencies = ['numpy>=1.9.0',
                        'scipy>=0.14.1',
                        'scikit-learn==0.18.1',
                        'pynisher==0.4.2',
                        'auto-sklearn==0.2.0']
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
        return info

    @AbstractBenchmark._check_configuration
    def objective_function(self, configuration, **kwargs):
        fold = kwargs['fold']
        folds = kwargs.get('folds', 10)
        cutoff = kwargs.get('cutoff', 1800)
        memory_limit = kwargs.get('memory_limit', 3072)
        subsample = kwargs.get('subsample', None)
        instance = json.dumps({'fold': fold, 'subsample': subsample})

        # (TODO) For now ignoring seed
        rng = kwargs.get("rng")
        self.rng = rng_helper.get_rng(rng=rng, self_rng=self.rng)

        if fold == folds:
            # run validation
            self.objective_function_test(configuration, cutoff=cutoff,
                                         memory_limit=memory_limit, rng=rng)

        include, _ = self._get_include_exclude_info()
        evaluator = autosklearn.evaluation.ExecuteTaFuncWithQueue(
            backend=self.backend,
            autosklearn_seed=1,
            resampling_strategy='partial-cv',
            folds=folds,
            logger=self.logger,
            memory_limit=memory_limit,
            metric=self.metric,
            include=include)

        status, cost, runtime, additional_run_info = evaluator.run(
            config=configuration, cutoff=cutoff, instance=instance)

        return {'function_value': cost, 'cost': runtime,
                'status': status, 'additional_run_info': additional_run_info}

    @AbstractBenchmark._check_configuration
    def objective_function_test(self, configuration, **kwargs):
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


class AutoSklearnBenchmarkAdultBAC(MulticlassClassificationBenchmark):
    def __init__(self, rng=None):
        super().__init__(2117, rng=rng)

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

        d["cvfolds"] = 10
        d["wallclocklimit"] = 24*60*60
        d['num_function_evals'] = np.inf
        d['cutoff'] = 1800
        return d
