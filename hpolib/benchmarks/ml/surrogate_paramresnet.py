import os
import pickle
import time
import numpy as np
import ConfigSpace as CS

from copy import deepcopy

from hpolib.abstract_benchmark import AbstractBenchmark
from hpolib.util.data_manager import SurrogateData

import sys
sys.path.append("/ihome/kleinaa/devel/git/lc_extrapolation")
from learning_curves_2 import MCMCCurveModelCombination


class SurrogateParamResNet(AbstractBenchmark):

    def __init__(self, dataset, path=None, rng=None):

        super(SurrogateParamResNet, self).__init__()

        url = ""
        surrogate = SurrogateData(surrogate_file=".pkl", url=url, folder="/")
        self.surrogate_objective = surrogate.load_objective()
        self.surrogate_cost = surrogate.load_cost()
        self.n_epochs = 200
        self.dataset = dataset
        if rng is None:
            self.rng = np.random.RandomState()
        else:
            self.rng = rng

    #@AbstractBenchmark._check_configuration  # Fails because of the internal order of the ConfigSpace
    @AbstractBenchmark._configuration_as_array
    def objective_function(self, x, step=None, **kwargs):
        x_ = deepcopy(x)
        x_[1] = 2 ** x_[1]  # batch size
        x_[3] = 10 ** x_[3]  # initial lr
        x_[4] = 10 ** x_[4]  # final lr
        x_[7] = 2 ** x_[7]  # num_units_1
        x_[10] = 2 ** x_[10]  # num_units_2
        x_[13] = 2 ** x_[13]  # num_units_3
        lc = self.surrogate_objective.predict(x_[None, :])[0]
        c = self.surrogate_cost.predict(x_[None, :])[0]

        if step is None:
            y = lc[-1]
            cost = c
        else:
            y = lc[step]
            cost = c / self.n_epochs * step

        return {'function_value': y, "cost": cost}

    #@AbstractBenchmark._check_configuration  # Fails because of the internal order of the ConfigSpace
    @AbstractBenchmark._configuration_as_array
    def objective_function_test(self, x, **kwargs):
        x_ = deepcopy(x)
        x_[1] = 2 ** x_[1]  # batch size
        x_[3] = 10 ** x_[3]  # initial lr
        x_[4] = 10 ** x_[4]  # final lr
        x_[7] = 2 ** x_[7]  # num_units_1
        x_[10] = 2 ** x_[10]  # num_units_2
        x_[13] = 2 ** x_[13]  # num_units_3
        lc = self.surrogate_objective.predict(x_[None, :])[0]
        c = self.surrogate_cost.predict(x_[None, :])[0]
        y = lc[-1]
        return {'function_value': y, "cost": c}

    @staticmethod
    def get_configuration_space():
        cs = CS.ConfigurationSpace()
        cs.generate_all_continuous_from_bounds(SurrogateParamResNet.get_meta_information()['bounds'])
        return cs

    @staticmethod
    def get_meta_information():
        return {'name': 'Residual Networks',

                # 'bounds': [[1.0, 3.0],  # num_stages
                #            [8.0, 256.0],  # batch_size
                #            [0.0, 0.9],  # momentum
                #            [1e-06, 1e-2],  # initial_lr
                #            [1e-4, 1],  # final_lr_fraction
                #            [1.0, 3.0],  # num_blocks_1
                #            [1.0, 5.0],  # num_layers_1
                #            [8.0, 256.0],  # num_units_1
                #            [1.0, 3.0],  # num_blocks_2
                #            [1.0, 5.0],  # num_layers_2
                #            [8.0, 256.0],  # num_units_2
                #            [1.0, 3.0],  # num_blocks_3
                #            [1.0, 5.0],  # num_layers_3
                #            [8.0, 256.0]]  # num_units_3
                'bounds': [[1.0, 3.0],  # num_stages
                           [3, 8],  # log 2 batch_size
                           [0, 1],  # momentum
                           [-6, -2],  # log 10 initial_lr
                           [-4, 0],  # log 10 final_lr_fraction
                           [1.0, 3.0],  # num_blocks_1
                           [1.0, 5.0],  # num_layers_1
                           [3, 8],  # log 2 num_units_1
                           [1.0, 3.0],  # num_blocks_2
                           [1.0, 5.0],  # num_layers_2
                           [3, 8],  # log 2 num_units_2
                           [1.0, 3.0],  # num_blocks_3
                           [1.0, 5.0],  # num_layers_3
                           [3, 8]]  # log 2 num_units_3
                }


class SurrogateParamResNetTime(SurrogateParamResNet):

    # @AbstractBenchmark._check_configuration  # Fails because of the internal order of the ConfigSpace
    @AbstractBenchmark._configuration_as_array
    def objective_function(self, x, budget=None, **kwargs):
        # If no budget is specified we train this config for the max number of epochs
        if budget is None:
            return super(SurrogateParamResNetTime, self).objective_function(x)
            
        x_ = deepcopy(x)
        x_[1] = 2 ** x_[1]  # batch size
        x_[3] = 10 ** x_[3]  # initial lr
        x_[4] = 10 ** x_[4]  # final lr
        x_[7] = 2 ** x_[7]  # num_units_1
        x_[10] = 2 ** x_[10]  # num_units_2
        x_[13] = 2 ** x_[13]  # num_units_3
        lc = self.surrogate_objective.predict(x_[None, :])[0]
        c = self.surrogate_cost.predict(x_[None, :])[0]

        # Check if we can afford a single epoch in the budget
        if c / self.n_epochs > budget:
            # TODO: Return random performance here instead
            y = 1
            return {'function_value': y, "cost": budget, "learning_curve": [y]}

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


class PredictiveTerminationCriterionResNet(SurrogateParamResNet):

    def __init__(self, interval, dataset, threshold=0.05, path=None, rng=None):
        super(PredictiveTerminationCriterionResNet, self).__init__(dataset, path=path, rng=rng)
        self.current_best_acc = -np.inf
        self.interval = interval
        self.threshold = threshold

        self.model = MCMCCurveModelCombination(100,
                                               nwalkers=100,
                                               nsamples=1000,
                                               burn_in=500,
                                               recency_weighting=False,
                                               soft_monotonicity_constraint=False,
                                               monotonicity_constraint=True,
                                               initial_model_weight_ml_estimate=True)

    # @AbstractBenchmark._check_configuration  # Fails because of the internal order of the ConfigSpace
    @AbstractBenchmark._configuration_as_array
    def objective_function(self, x, step=None, **kwargs):
        start_time = time.time()
        lc = []

        for i in range(self.n_epochs):
            res = super(PredictiveTerminationCriterionResNet, self).objective_function(x, step=i)
            lc.append(1 - res["function_value"])

            if i > 0 and i % self.interval == 0:

                # Fit learning curve model
                t_idx = np.arange(1, len(lc) + 1)
                self.model.fit(t_idx, lc)

                p_greater = self.model.posterior_prob_x_greater_than(self.n_epochs + 1, self.current_best_acc)
                print(i, p_greater)
                if p_greater >= self.threshold:
                    continue
                else:
                    m = np.mean(self.model.predictive_distribution(self.n_epochs + 1))
                    c = time.time() - start_time + res["cost"]

                    print(m, c)

                    return {'function_value': 1 - m, "cost": c, 'observed_epochs': i}

        c = time.time() - start_time + res["cost"]
        print(lc[-1], self.current_best_acc)
        if lc[-1] > self.current_best_acc:
            self.current_best_acc = lc[-1]

        return {'function_value': 1 - lc[-1], "cost": c, 'observed_epochs': self.n_epochs}


class PredictiveTerminationCriterionResNetTime(SurrogateParamResNetTime):

    def __init__(self, n_steps, dataset, threshold=0.05, path=None, rng=None):
        super(PredictiveTerminationCriterionResNetTime, self).__init__(dataset, path=path, rng=rng)
        self.current_best_acc = -np.inf
        self.n_steps = n_steps
        self.threshold = threshold

        self.model = MCMCCurveModelCombination(100,
                                               nwalkers=100,
                                               nsamples=1000,
                                               burn_in=500,
                                               recency_weighting=False,
                                               soft_monotonicity_constraint=False,
                                               monotonicity_constraint=True,
                                               initial_model_weight_ml_estimate=True)

    @AbstractBenchmark._configuration_as_array
    def objective_function(self, x, budget=None, **kwargs):

        time_steps = budget / self.n_steps

        for i in range(1, self.n_steps):

            res = super(PredictiveTerminationCriterionResNetTime, self).objective_function(x, budget=(time_steps * i))

            lc = [1 - l for l in res["learning_curve"]]

            # Fit learning curve model
            t_idx = np.arange(1, len(lc) + 1)
            self.model.fit(t_idx, lc)

            print(lc[-1], self.current_best_acc)

            p_greater = self.model.posterior_prob_x_greater_than(self.n_epochs + 1, self.current_best_acc)
            print(p_greater, i)
            if p_greater >= self.threshold:
                continue
            else:
                m = np.mean(self.model.predictive_distribution(self.n_epochs + 1))
                print("Killed", m, self.current_best_acc)
                return {'function_value': 1 - m, "cost": time_steps * i, 'observed_epochs': len(lc)}

        res = super(PredictiveTerminationCriterionResNetTime, self).objective_function(x, budget=budget)

        if (1 - res["function_value"]) > self.current_best_acc:
            self.current_best_acc = (1 - res["function_value"])

        return res