import time

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import preprocessing, pipeline
from sklearn.neural_network import MLPClassifier

import ConfigSpace as CS

from hpolib.abstract_benchmark import AbstractBenchmark
from hpolib.util.openml_data_manager import OpenMLHoldoutDataManager
import hpolib.util.rng_helper as rng_helper
from hpolib.util.data_manager import MNISTData


class MLP(AbstractBenchmark):

    def __init__(self, max_iterations=100, n_threads=1, rng=None):
        """
        Multi-Layer Perceptron as implemented in scikit-learn
            http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
        Parameters
        ----------
        rng: str
            set up rng
        """
        super(MLP, self).__init__(rng=rng)

        self.X_trn, self.y_trn, self.X_val, self.y_val, self.X_tst, self.y_tst = self.get_data()
        self.max_iterations = max_iterations
        self.n_threads = n_threads

    def get_data(self):
        raise NotImplementedError("Do not use this class as this is only a skeleton for further implementations.")

    @AbstractBenchmark._check_configuration
    @AbstractBenchmark._configuration_as_array
    def objective_function(self, x, **kwargs):
        l2_reg = 10 ** x[0]
        momentum = x[1]
        init_lr = 10 ** x[2]
        power_t = x[3]

        rng = kwargs.get("rng", None)
        self.rng = rng_helper.get_rng(rng=rng, self_rng=self.rng)
        start = time.time()
        model = self._get_pipeline(l2_reg=l2_reg, momentum=momentum, init_lr=init_lr, power_t=power_t)
        model.fit(X=self.X_trn.copy(), y=self.y_trn.copy())
        val_loss = self._test_model(model=model, X=self.X_val, y=self.y_val)
        cost = time.time() - start
        return {'function_value': val_loss, "cost": cost}

    @AbstractBenchmark._check_configuration
    @AbstractBenchmark._configuration_as_array
    def objective_function_test(self, x, **kwargs):
        l2_reg = 10**x[0]
        momentum = x[1]
        init_lr = 10**x[2]
        power_t = x[3]

        rng = kwargs.get("rng", None)
        self.rng = rng_helper.get_rng(rng=rng, self_rng=self.rng)
        start = time.time()
        model = self._get_pipeline(l2_reg=l2_reg, momentum=momentum, init_lr=init_lr, power_t=power_t)
        model.fit(X=np.concatenate((self.X_trn, self.X_val)),
                  y=np.concatenate((self.y_trn, self.y_val)),)
        tst_loss = self._test_model(model=model, X=self.X_tst, y=self.y_tst)
        cost = time.time() - start
        return {'function_value': tst_loss, "cost": cost}

    @staticmethod
    def get_configuration_space():
        cs = CS.ConfigurationSpace(seed=1)
        cs.generate_all_continuous_from_bounds(MLP.get_meta_information()['bounds'])
        return cs

    @staticmethod
    def get_meta_information():
        return {'name': 'MLP',
                'bounds': [[-8.0, -1.0],  # l2_reg
                           [0.3, 0.999],  # momentum
                           [-6.0, 0.0],   # init_learning_rate
                           [0.0, 1.0],    # exponent for inv lr schedule
                           ],
                'num_function_evals': 50,
                'requires': ["scikit-learn", ],
                'references': ["None"]
                }

    def _get_pipeline(self, l2_reg, init_lr, momentum, power_t):
        clf = pipeline.Pipeline([('preproc', preprocessing.StandardScaler()),
                                 ('class', MLPClassifier(hidden_layer_sizes=(300, 300),
                                                         activation='relu',
                                                         solver='sgd',
                                                         alpha=l2_reg,
                                                         batch_size='auto',
                                                         learning_rate='invscaling',
                                                         learning_rate_init=init_lr,
                                                         momentum=momentum,
                                                         power_t=power_t,
                                                         nesterovs_momentum=True,
                                                         early_stopping=True,
                                                         validation_fraction=0.2,
                                                         max_iter=200,
                                                         shuffle=True,
                                                         random_state=self.rng))
                                 ])
        return clf

    def _test_model(self, X, y, model):
        y_pred = model.predict(X)
        acc = accuracy_score(y_pred=y_pred, y_true=y)
        return 1-acc


class MLPOnHiggs(MLP):

    def get_data(self):
        dm = OpenMLHoldoutDataManager(openml_task_id=75101, rng=self.rng)
        X_train, y_train, X_val, y_val, X_test, y_test = dm.load()
        return X_train, y_train, X_val, y_val, X_test, y_test

    def _get_pipeline(self, l2_reg, init_lr, momentum, power_t):
        clf = pipeline.Pipeline([('preproc1', preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)),
                                 ('preproc2', preprocessing.StandardScaler()),
                                 ('classifier', MLPClassifier(hidden_layer_sizes=(300, 300),
                                                              activation='relu',
                                                              solver='sgd',
                                                              alpha=l2_reg,
                                                              batch_size='auto',
                                                              learning_rate='invscaling',
                                                              learning_rate_init=init_lr,
                                                              momentum=momentum,
                                                              power_t=power_t,
                                                              nesterovs_momentum=True,
                                                              early_stopping=True,
                                                              validation_fraction=0.2,
                                                              max_iter=200,
                                                              shuffle=True,
                                                              random_state=self.rng))
                                 ])
        return clf

    @staticmethod
    def get_meta_information():
        d = MLP.get_meta_information()
        data_ref = ["@inproceedings{baldi-nature14,"
                    "author = {P. Baldi and P. Sadowski and D. Whiteson},"
                    "year = {2014},"
                    "month = {07},"
                    "pages = {1-9},"
                    "title = {Searching for Exotic Particles in High-Energy Physics with Deep Learning},"
                    "volume = {5},"
                    "booktitle = {Nature Communications}"
                    ]
        d["references"].append(data_ref)
        return d


class MLPOnMnist(MLP):

    def get_data(self):
        dm = MNISTData()
        return dm.load()

    @staticmethod
    def get_meta_information():
        d = MLP.get_meta_information()
        data_ref = ["@inproceedings{lecun-ieee98,"
                    "title={Gradient-based learning applied to document recognition},"
                    "author={Y. LeCun and L. Bottou and Y. Bengio and P. Haffner},"
                    "journal={Proceedings of the IEEE},"
                    "pages={2278--2324},"
                    "year={1998},"
                    "publisher={IEEE}"]
        d["references"].append(data_ref)
        return d

class MLPOnVehicle(MLP):

    def get_data(self):
        dm = OpenMLHoldoutDataManager(openml_task_id=75191, rng=self.rng)
        return dm.load()
