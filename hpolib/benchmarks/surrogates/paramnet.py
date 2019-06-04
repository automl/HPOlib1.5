import numpy as np
import ConfigSpace as CS

from copy import deepcopy

from hpolib.abstract_benchmark import AbstractBenchmark
from hpolib.benchmarks.surrogates.surrogate_benchmark import SurrogateBenchmark


class SurrogateParamNet(SurrogateBenchmark):

    def __init__(self, dataset, **kwargs):
        """

        :param dataset: Dataset the network was trained on. Available: adult, higgs, letter, mnist, optdigits, poker

        :param kwargs:
        """
        if dataset not in ("adult", "higgs", "letter", "mnist", "optdigits", "poker"):
            raise ValueError("No surrogate found for %s" % dataset)

        self.n_epochs = 50
        self.dataset = dataset
        
        objective_fn = "rf_surrogate_paramnet_%s.pkl" % dataset
        cost_fn = "rf_cost_surrogate_paramnet_%s.pkl" % dataset

        super(SurrogateParamNet, self).__init__(objective_surrogate_fn=objective_fn, cost_surrogate_fn=cost_fn, **kwargs)

    def get_empirical_f_opt(self):
        """
        Returns average across minimal value in all trees
        :return:
        """
        ms = []
        for t in self.surrogate_objective.estimators_:
            ms.append(np.min(t.tree_.value))
        return np.mean(ms)

    @AbstractBenchmark._check_configuration
    @AbstractBenchmark._configuration_as_array
    def objective_function(self, x, step=None, **kwargs):
        x_ = deepcopy(x)
        x_[0] = 10 ** x_[0]
        x_[1] = 2 ** x_[1]
        x_[2] = 2 ** x_[2]
        x_[3] = 10 ** x_[3]
        lc = self.surrogate_objective.predict(x_[None, :])[0]
        c = self.surrogate_cost.predict(x_[None, :])[0]

        if step is None:
            y = lc[-1]
            cost = c
        else:
            y = lc[step]
            cost = c / self.n_epochs * step

        return {'function_value': y, "cost": cost, "learning_curve": lc}

    @AbstractBenchmark._check_configuration
    @AbstractBenchmark._configuration_as_array
    def objective_function_test(self, x, **kwargs):
        x_ = deepcopy(x)
        x_[0] = 10 ** x_[0]
        x_[1] = 2 ** x_[1]
        x_[2] = 2 ** x_[2]
        x_[3] = 10 ** x_[3]
        lc = self.surrogate_objective.predict(x_[None, :])[0]
        c = self.surrogate_cost.predict(x_[None, :])[0]
        y = lc[-1]
        return {'function_value': y, "cost": c}

    @staticmethod
    def get_configuration_space():
        cs = CS.ConfigurationSpace()
        cs.generate_all_continuous_from_bounds(SurrogateParamNet.get_meta_information()['bounds'])
        return cs

    @staticmethod
    def get_meta_information():
        return {'name': 'Stefans reparameterization of paramnet',
                # 'bounds': [[1e-6, 1e-2],  # initial_lr
                #            [8, 256],  # batch_size
                #            [16, 256],  # average_units_per_layer
                #            [1e-4, 1],  # final_lr_fraction
                #            [0, 1.],  # shape_parameter_1
                #            [1, 5],  # num_layers
                #            [0, .5],  # dropout_0
                #            [0, .5]]  # dropout_1
                'bounds': [[-6, -2],  # log10 initial_lr
                           [3, 8],  # log2 batch_size
                           [4, 8],  # log2 average_units_per_layer
                           [-4, 0],  # log10 final_lr_fraction
                           [0, 1.],  # shape_parameter_1
                           [1, 5],  # num_layers
                           [0, .5],  # dropout_0
                           [0, .5]]  # dropout_1
                }


class SurrogateReducedParamNetTime(SurrogateParamNet):

    @AbstractBenchmark._check_configuration
    @AbstractBenchmark._configuration_as_array
    def objective_function(self, x, budget=None, **kwargs):
        # If no budget is specified we train this config for the max number of epochs
        if budget is None:
            budget = np.inf

        x_ = np.zeros([1, 8], dtype=np.float)
        
        x_[0, 0] = 10 ** x[0]
        x_[0, 1] = 2 ** x[1]
        x_[0, 2] = 2 ** x[2]
        x_[0, 3] = 10 ** x[3]
        x_[0, 4] = 0.5
        x_[0, 5] = x[4]
        x_[0, 6] = x[5]
        x_[0, 7] = x[5]

        lc = self.surrogate_objective.predict(x_)[0]
        c = self.surrogate_cost.predict(x_)[0]

        # Check if we can afford a single epoch in the budget
        if c / self.n_epochs > budget:
            # TODO: Return random performance here instead
            y = 1
            return {'function_value': y, "cost": budget}

        learning_curves_cost = np.linspace(c / self.n_epochs, c, self.n_epochs)

        idx = np.where(learning_curves_cost < budget)[0][-1]
        y = lc[idx]

        return {'function_value': y, "cost": budget, "lc": lc[:idx].tolist()}

    @AbstractBenchmark._check_configuration
    @AbstractBenchmark._configuration_as_array
    def objective_function_test(self, x, **kwargs):
        x_ = np.zeros([1, 8], dtype=np.float)

        x_[0, 0] = 10 ** x[0]
        x_[0, 1] = 2 ** x[1]
        x_[0, 2] = 2 ** x[2]
        x_[0, 3] = 10 ** x[3]
        x_[0, 4] = 0.5
        x_[0, 5] = x[4]
        x_[0, 6] = x[5]
        x_[0, 7] = x[5]
        lc = self.surrogate_objective.predict(x_)[0]
        c = self.surrogate_cost.predict(x_)[0]
        y = lc[-1]
        return {'function_value': y, "cost": c}

    @staticmethod
    def get_configuration_space():
        cs = CS.ConfigurationSpace()
        cs.generate_all_continuous_from_bounds(SurrogateReducedParamNetTime.get_meta_information()['bounds'])
        return cs

    @staticmethod
    def get_meta_information():
        return {'name': 'Stefans reparameterization of paramnet',
                # 'bounds': [[1e-6, 1e-2],  # initial_lr
                #            [8, 256],  # batch_size
                #            [16, 256],  # average_units_per_layer
                #            [1e-4, 1],  # final_lr_fraction
                #            [0, 1.],  # shape_parameter_1
                #            [1, 5],  # num_layers
                #            [0, .5],  # dropout_0
                #            [0, .5]]  # dropout_1
                'bounds': [[-6, -2],  # log10 initial_lr
                           [3, 8],  # log2 batch_size
                           [4, 8],  # log2 average_units_per_layer
                           [-4, 0],  # log10 final_lr_fraction
                           #[0, 1.],  # shape_parameter_1
                           [1, 5],  # num_layers
                           #[0, .5]  # dropout_0
                           [0, .5]]  # dropout_1
                }


class SurrogateParamNetTime(SurrogateParamNet):

    @AbstractBenchmark._check_configuration
    @AbstractBenchmark._configuration_as_array
    def objective_function(self, x, budget=None, **kwargs):
        # If no budget is specified we train this config for the max number of epochs
        if budget is None:
            return super(SurrogateParamNetTime, self).objective_function(x)
            
        x_ = deepcopy(x)
        x_[0] = 10 ** x_[0]
        x_[1] = 2 ** x_[1]
        x_[2] = 2 ** x_[2]
        x_[3] = 10 ** x_[3]
        lc = self.surrogate_objective.predict(x_[None, :])[0]
        c = self.surrogate_cost.predict(x_[None, :])[0]

        # Check if we can afford a single epoch in the budget
        if c / self.n_epochs > budget:
            # TODO: Return random performance here instead
            y = 1
            return {'function_value': y, "cost": budget}

        learning_curves_cost = np.linspace(c / self.n_epochs, c, self.n_epochs)

        if budget < c:
            idx = np.where(learning_curves_cost < budget)[0][-1]
            y = lc[idx]
            return {'function_value': y, "cost": budget, "learning_curve": lc[:idx], 'observed_epochs': len(lc[:idx])}
        else:
            # If the budget is larger than the actual runtime we extrapolate the learning curve
            t_left = budget - c
            n_epochs = int(t_left / (c / self.n_epochs))
            lc = np.append(lc, np.ones(n_epochs) * lc[-1])
            y = lc[-1]
            return {'function_value': y, "cost": budget, "learning_curve": lc, 'observed_epochs': len(lc)}
