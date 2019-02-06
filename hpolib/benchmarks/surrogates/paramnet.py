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

        self.n_epochs = 50
        self.dataset = dataset
        
        objective_fn = "rf_surrogate_paramnet_%s.pkl" % dataset
        cost_fn = "rf_cost_surrogate_paramnet_%s.pkl" % dataset

        super(SurrogateParamNet, self).__init__(objective_surrogate_fn=objective_fn, cost_surrogate_fn=cost_fn, **kwargs)


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
        x_ = deepcopy(x)
        x_[0] = 10 ** x_[0]
        x_[1] = 2 ** x_[1]
        x_[2] = 2 ** x_[2]
        x_[3] = 10 ** x_[3]
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


# class PredictiveTerminationCriterion(SurrogateParamNet):
#
#     def __init__(self, interval, dataset, threshold=0.05, path=None, rng=None):
#         super(PredictiveTerminationCriterion, self).__init__(dataset, path=path, rng=rng)
#         self.current_best_acc = -np.inf
#         self.interval = interval
#         self.threshold = threshold
#
#         # self.model = setup_model_combination(self.n_epochs + 1,
#         #                                 models=['weibull', 'pow4', 'mmf', 'pow3',
#         #                                         'loglog_linear', 'janoschek',
#         #                                         'dr_hill_zero_background', 'log_power', 'exp4'],
#         #                                 monotonicity_constraint=False,
#         #                                 soft_monotonicity_constraint=False)
#         self.model = MCMCCurveModelCombination(100,
#                                                nwalkers=100,
#                                                nsamples=1000,
#                                                burn_in=500,
#                                                recency_weighting=False,
#                                                soft_monotonicity_constraint=False,
#                                                monotonicity_constraint=True,
#                                                initial_model_weight_ml_estimate=True)
#
#     @AbstractBenchmark._check_configuration
#     @AbstractBenchmark._configuration_as_array
#     def objective_function(self, x, step=None, **kwargs):
#         start_time = time.time()
#         lc = []
#
#         for i in range(self.n_epochs):
#             res = super(PredictiveTerminationCriterion, self).objective_function(x, step=i)
#             lc.append(1 - res["function_value"])
#
#             if i > 0 and i % self.interval == 0:
#
#                 # Fit learning curve model
#                 t_idx = np.arange(1, len(lc) + 1)
#                 self.model.fit(t_idx, lc)
#
#                 p_greater = self.model.posterior_prob_x_greater_than(self.n_epochs + 1, self.current_best_acc)
#
#                 if p_greater >= self.threshold:
#                     continue
#                 else:
#                     m = np.mean(self.model.predictive_distribution(self.n_epochs + 1))
#                     c = time.time() - start_time + res["cost"]
#
#                     return {'function_value': 1 - m, "cost": c, 'observed_epochs': i}
#
#         c = time.time() - start_time + res["cost"]
#
#         if lc[-1] > self.current_best_acc:
#             self.current_best_acc = lc[-1]
#
#         return {'function_value': 1 - lc[-1], "cost": c, 'observed_epochs': self.n_epochs}
#
#
# class PredictiveTerminationCriterionParamNetTime(SurrogateParamNetTime):
#
#     def __init__(self, n_steps, dataset, threshold=0.05, path=None, rng=None):
#         super(PredictiveTerminationCriterionParamNetTime, self).__init__(dataset, path=path, rng=rng)
#         self.current_best_acc = -np.inf
#         self.n_steps = n_steps
#         self.threshold = threshold
#
#         self.model = MCMCCurveModelCombination(100,
#                                                nwalkers=100,
#                                                nsamples=1000,
#                                                burn_in=500,
#                                                recency_weighting=False,
#                                                soft_monotonicity_constraint=False,
#                                                monotonicity_constraint=True,
#                                                initial_model_weight_ml_estimate=True)
#
#     @AbstractBenchmark._configuration_as_array
#     def objective_function(self, x, budget=None, **kwargs):
#
#         time_steps = budget / self.n_steps
#
#         for i in range(1, self.n_steps):
#
#             res = super(PredictiveTerminationCriterionParamNetTime, self).objective_function(x, budget=(time_steps * i))
#
#             lc = [1 - l for l in res["learning_curve"]]
#
#             # Fit learning curve model
#             t_idx = np.arange(1, len(lc) + 1)
#             self.model.fit(t_idx, lc)
#
#             print(lc[-1], self.current_best_acc)
#
#             p_greater = self.model.posterior_prob_x_greater_than(self.n_epochs + 1, self.current_best_acc)
#             print(p_greater, i)
#             if p_greater >= self.threshold:
#                 continue
#             else:
#                 m = np.mean(self.model.predictive_distribution(self.n_epochs + 1))
#                 print("Killed", m, self.current_best_acc)
#                 return {'function_value': 1 - m, "cost": time_steps * i, 'observed_epochs': len(lc)}
#
#         res = super(PredictiveTerminationCriterionParamNetTime, self).objective_function(x, budget=budget)
#
#         if (1 - res["function_value"]) > self.current_best_acc:
#             self.current_best_acc = (1 - res["function_value"])
#
#         return res
