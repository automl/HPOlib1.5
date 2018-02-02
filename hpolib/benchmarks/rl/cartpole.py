import time
import numpy as np
import ConfigSpace as CS

from tensorforce.agents import PPOAgent
from tensorforce.execution import Runner
from tensorforce.contrib.openai_gym import OpenAIGym

from hpolib.abstract_benchmark import AbstractBenchmark
from hpolib.util import rng_helper


class Cartpole(AbstractBenchmark):
    """

    """

    def __init__(self, rng=None):
        """
        Parameters
        ----------
        rng: int/None/RandomState
            set up rng
        """

        super(Cartpole, self).__init__()

        self.rng = rng_helper.create_rng(rng)
        self.env = OpenAIGym('CartPole-v0', visualize=False)
        self.max_episodes = 3000
        self.avg_n_episodes = 20
        self.max_budget = 9

    @AbstractBenchmark._check_configuration
    def objective_function(self, config, budget=None, **kwargs):

        st = time.time()

        if budget is None:
            budget = self.max_budget

        network_spec = [
            dict(type='dense', size=config["n_units_1"], activation=config['activation_1']),
            dict(type='dense', size=config["n_units_2"], activation=config['activation_2'])
        ]

        network_spec_baseline = [
            dict(type='dense', size=config["baseline_n_units_1"], activation=config['baseline_activation_1']),
            dict(type='dense', size=config["baseline_n_units_2"], activation=config['baseline_activation_2'])
        ]

        converged_episodes = []

        for i in range(budget):
            agent = PPOAgent(
                states_spec=self.env.states,
                actions_spec=self.env.actions,
                network_spec=network_spec,
                batch_size=config["batch_size"],
                # Agent
                states_preprocessing_spec=None,
                explorations_spec=None,
                reward_preprocessing_spec=None,
                # BatchAgent
                keep_last_timestep=True,
                # PPOAgent
                step_optimizer=dict(
                    type='adam',
                    learning_rate=config["learning_rate"]
                ),
                optimization_steps=config["optimization_steps"],
                # Model
                scope='ppo',
                discount=config["discount"],
                # DistributionModel
                distributions_spec=None,
                entropy_regularization=config["entropy_regularization"],
                # PGModel
                baseline_mode=config["baseline_mode"],
                baseline=network_spec_baseline,
                baseline_optimizer=dict(
                    type='adam',
                    learning_rate=config["baseline_learning_rate"]
                ),
                gae_lambda=None,
                # PGLRModel
                likelihood_ratio_clipping=config["likelihood_ratio_clipping"],
                summary_spec=None,
                distributed_spec=None
            )

            def episode_finished(r):
                # Check if we have converged
                if np.mean(r.episode_rewards[-self.avg_n_episodes:]) == 200:
                    return False
                else:
                    return True

            runner = Runner(agent=agent, environment=self.env)

            runner.run(episodes=self.max_episodes, max_episode_timesteps=200, episode_finished=episode_finished)

            converged_episodes.append(len(runner.episode_rewards))

        cost = time.time() - st

        return {'function_value': np.mean(converged_episodes), "cost": cost, "all_runs": converged_episodes}

    @AbstractBenchmark._check_configuration
    def objective_function_test(self, config, **kwargs):

        return self.objective_function(self, config, budget=self.max_budget)

    @staticmethod
    def get_configuration_space():
        cs = CS.ConfigurationSpace()

        cs.add_hyperparameter(CS.UniformIntegerHyperparameter("n_units_1",
                                                              lower=8,
                                                              default_value=64,
                                                              upper=128,
                                                              log=True))

        cs.add_hyperparameter(CS.UniformIntegerHyperparameter("n_units_2",
                                                              lower=8,
                                                              default_value=64,
                                                              upper=128,
                                                              log=True))

        cs.add_hyperparameter(CS.UniformIntegerHyperparameter("batch_size",
                                                              lower=8,
                                                              default_value=64,
                                                              upper=256,
                                                              log=True))

        cs.add_hyperparameter(CS.UniformFloatHyperparameter("learning_rate",
                                                            lower=1e-7,
                                                            default_value=1e-3,
                                                            upper=1e-1,
                                                            log=True))

        cs.add_hyperparameter(CS.UniformFloatHyperparameter("discount",
                                                            lower=0,
                                                            default_value=.99,
                                                            upper=1))

        cs.add_hyperparameter(CS.UniformFloatHyperparameter("likelihood_ratio_clipping",
                                                            lower=0,
                                                            default_value=.2,
                                                            upper=1))

        cs.add_hyperparameter(CS.CategoricalHyperparameter("activation_1", ["tanh", "relu"]))

        cs.add_hyperparameter(CS.CategoricalHyperparameter("activation_2", ["tanh", "relu"]))

        cs.add_hyperparameter(CS.UniformIntegerHyperparameter("optimization_steps",
                                                              lower=1,
                                                              default_value=10,
                                                              upper=100))

        cs.add_hyperparameter(CS.CategoricalHyperparameter("baseline_mode", ["states", "network"]))

        cs.add_hyperparameter(CS.UniformIntegerHyperparameter("baseline_n_units_1",
                                                              lower=8,
                                                              default_value=64,
                                                              upper=128,
                                                              log=True))

        cs.add_hyperparameter(CS.UniformIntegerHyperparameter("baseline_n_units_2",
                                                              lower=8,
                                                              default_value=64,
                                                              upper=128,
                                                              log=True))

        cs.add_hyperparameter(CS.CategoricalHyperparameter("baseline_activation_1", ["tanh", "relu"]))

        cs.add_hyperparameter(CS.CategoricalHyperparameter("baseline_activation_2", ["tanh", "relu"]))

        cs.add_hyperparameter(CS.UniformFloatHyperparameter("baseline_learning_rate",
                                                            lower=1e-7,
                                                            default_value=1e-3,
                                                            upper=1e-1,
                                                            log=True))

        return cs

    @staticmethod
    def get_meta_information():
        return {'name': 'Cartpole',
                'references': []
                }
