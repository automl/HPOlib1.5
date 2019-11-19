import time
import logging

import ConfigSpace as CS
import numpy as np
from hpolib.abstract_benchmark import AbstractBenchmark
from hpolib.util import rng_helper
try:
    from tensorforce.agents import PPOAgent
    from tensorforce.contrib.openai_gym import OpenAIGym
    from tensorforce.execution import Runner
except ImportError:
    raise ImportError(
        "Missing tensorforce dependency. "
        "Install extra 'rl': pip install hpolib2[rl]==0.0.1 .")


__version__ = "0.2"


class ClassicControlBase(AbstractBenchmark):
    def __init__(self, env="CartPole-v1", rng=None, defaults=None, max_budget=9,
                 avg_n_episodes=20, max_episodes=3000, max_timesteps=None):
        """
        Parameters
        ----------
        rng: int/None/RandomState
            set up rng
        defaults: dict
            default configuration used for the PPO agent
        avg_n_episodes: Optional[int]
            number of episodes of which average reward is above threshold for convergence
        max_episodes: int
            maximum number of episodes to run
        """

        super(ClassicControlBase, self).__init__()

        self.logger = logging.getLogger(self.__class__.__name__)
        self.rng = rng_helper.create_rng(rng)

        allowed_envs = ["CartPole-v1", "Acrobot-v1", "MountainCar-v0", "Pendulum-v0", "MountainCarContinuous-v0"]
        assert env in allowed_envs, "Expected env in {}, got {}".format(", ".join(allowed_envs), env)

        self.env = OpenAIGym(env, visualize=False)
        self.max_episodes = max_episodes
        self.max_timesteps = max_timesteps
        self.avg_n_episodes = avg_n_episodes
        self.max_budget = max_budget
        self.defaults = {
            "n_units_1": 64,
            "n_units_2": 64,
            "batch_size": 64,
            "learning_rate": 1e-3,
            "discount": 0.99,
            "likelihood_ratio_clipping": 0.2,
            "activation_1": "tanh",
            "activation_2": "tanh",
            "optimizer_type": "adam",
            "optimization_steps": 10,

            "baseline_mode": "states",
            "baseline_n_units_1": 64,
            "baseline_n_units_2": 64,
            "baseline_learning_rate": 1e-3,
            "baseline_optimization_steps": 10,
            "baseline_optimizer_type": "adam"
        }

        if defaults is not None:
            self.defaults.update(defaults)

    @AbstractBenchmark._check_configuration
    def objective_function(self, config, budget=None, **kwargs):

        # fill in missing entries with default values for 'incomplete/reduced' configspaces
        c = self.defaults
        c.update(config)
        config = c

        st = time.time()

        if budget is None:
            budget = self.max_budget

        network_spec = [
            dict(type='dense', size=config["n_units_1"], activation=config['activation_1']),
            dict(type='dense', size=config["n_units_2"], activation=config['activation_2'])
        ]

        terminated_runs = []

        for i in range(budget):
            agent = PPOAgent(
                states=self.env.states,
                actions=self.env.actions,
                network=network_spec,

                update_mode=dict(unit='episodes',
                                 batch_size=config["batch_size"]),

                step_optimizer=dict(
                    type=config["optimizer_type"],
                    learning_rate=config["learning_rate"]
                ),
                optimization_steps=config["optimization_steps"],

                discount=config["discount"],

                baseline_mode=config["baseline_mode"],
                baseline={
                    "type": "mlp",
                    "sizes": [config["baseline_n_units_1"], config["baseline_n_units_2"]]
                },
                baseline_optimizer={
                    "type": "multi_step",
                    "optimizer": {
                        "type": config["baseline_optimizer_type"],
                        "learning_rate": config["baseline_learning_rate"]
                    },
                    "num_steps": config["baseline_optimization_steps"]
                },
                likelihood_ratio_clipping=config["likelihood_ratio_clipping"],
            )

            runner = Runner(agent=agent, environment=self.env)
            runner.run(num_episodes=self.max_episodes, num_timesteps=self.max_timesteps,
                       episode_finished=self._on_episode_finished)

            reward_per_episode = np.mean(runner.episode_rewards)
            if self.max_episodes:  # converged, interpolate leftover episodes
                interpolated_leftover_episodes = self._interpolate_reward(
                    runner, reward_per_episode, runner.global_episode, self.max_episodes)
            else:
                interpolated_leftover_episodes = reward_per_episode
            if self.max_timesteps:  # converged, interpolate leftover timesteps
                interpolated_leftover_timesteps = self._interpolate_reward(
                    runner, reward_per_episode, runner.global_timestep, self.max_timesteps)
            else:
                interpolated_leftover_timesteps = reward_per_episode
            terminated_runs.append(max(reward_per_episode,
                                       min(interpolated_leftover_episodes, interpolated_leftover_timesteps)))

        cost = time.time() - st
        negative_mean_reward = -np.mean(terminated_runs)  # minimization task
        return {'function_value': negative_mean_reward, "cost": cost, "all_runs": terminated_runs}

    def _on_episode_finished(self, runner, worker_id):
        if runner.global_episode % 10 == 0:
            self.logger.info("Finished episode {} at timestep {}. Avg recent episodes: reward {:.1f}, timesteps {:.1f}"
                             .format(runner.global_episode, runner.global_timestep,
                                     np.mean(runner.episode_rewards[-10:]), np.mean(runner.episode_timesteps[-10:])))
        if self.avg_n_episodes is None:  # no convergence criterion
            converged = False
        else:
            mean_reward = np.mean(runner.episode_rewards[-self.avg_n_episodes:])
            converged = mean_reward >= self._reward_threshold()
        return not converged

    _fallback_reward_thresholds = {  # according to gym 0.12.5
        'Acrobot-v1': 500,
        'Pendulum-v0': 200,
    }

    def _reward_threshold(self):
        """ Shim reward thresholds not available
            in gym==0.9.5 (required by tensorforce)
        """
        threshold = self.env.gym.unwrapped.spec.reward_threshold
        if threshold is None:
            return self._fallback_reward_thresholds[self.env.gym_id]
        else:
            return threshold

    def _interpolate_reward(self, runner, current_reward_per_episode, current_time, max_time):
        leftover_time = max_time - current_time
        if leftover_time <= 0:
            return current_reward_per_episode
        converged_reward_per_episode = max(np.mean(runner.episode_rewards[-self.avg_n_episodes:]),
                                           self._reward_threshold())
        return (current_reward_per_episode * current_time + converged_reward_per_episode * leftover_time) / max_time

    @AbstractBenchmark._check_configuration
    def objective_function_test(self, config, **kwargs):
        return self.objective_function(config, budget=self.max_budget, **kwargs)

    @staticmethod
    def get_meta_information():
        return {'name': 'ClassicControl with PPO',
                'references': [],
                'note': ("Version 0.2:\n"
                         "- Predefine reward thresholds of convergence criteria for the particular environment\n"
                         "- Use max episode timesteps predefined by the particular environment\n"
                         "- Use mean of episode rewards as run performance\n"
                         "- Log mean performance every 10th episode\n"
                         "- Command line interface for running default configuration\n"
                         "- Stop at maximum episodes OR maximum timesteps")
                }


class ClassicControlFull(ClassicControlBase):

    @staticmethod
    def get_configuration_space():
        cs = CS.ConfigurationSpace()

        cs.add_hyperparameter(CS.UniformIntegerHyperparameter(
            "n_units_1", lower=8, default_value=64, upper=64, log=True))
        cs.add_hyperparameter(CS.UniformIntegerHyperparameter(
            "n_units_2", lower=8, default_value=64, upper=64, log=True))
        cs.add_hyperparameter(CS.UniformIntegerHyperparameter(
            "batch_size", lower=8, default_value=64, upper=256, log=True))
        cs.add_hyperparameter(CS.UniformFloatHyperparameter(
            "learning_rate", lower=1e-7, default_value=1e-3, upper=1e-1, log=True))
        cs.add_hyperparameter(CS.UniformFloatHyperparameter(
            "discount", lower=0, default_value=.99, upper=1))
        cs.add_hyperparameter(CS.UniformFloatHyperparameter(
            "likelihood_ratio_clipping", lower=1e-3, default_value=.2, upper=1))
        cs.add_hyperparameter(CS.CategoricalHyperparameter(
            "activation_1", ["tanh", "relu"]))
        cs.add_hyperparameter(CS.CategoricalHyperparameter(
            "activation_2", ["tanh", "relu"]))
        cs.add_hyperparameter(CS.CategoricalHyperparameter(
            "optimizer_type", ["adam", "rmsprop"]))
        cs.add_hyperparameter(CS.UniformIntegerHyperparameter(
            "optimization_steps", lower=1, default_value=10, upper=10))
        cs.add_hyperparameter(CS.CategoricalHyperparameter(
            "baseline_mode", ["states", "network"]))
        cs.add_hyperparameter(CS.UniformIntegerHyperparameter(
            "baseline_n_units_1", lower=8, default_value=64, upper=128, log=True))
        cs.add_hyperparameter(CS.UniformIntegerHyperparameter(
            "baseline_n_units_2", lower=8, default_value=64, upper=128, log=True))
        cs.add_hyperparameter(CS.UniformFloatHyperparameter(
            "baseline_learning_rate", lower=1e-7, default_value=1e-3, upper=1e-1, log=True))
        cs.add_hyperparameter(CS.UniformIntegerHyperparameter(
            "baseline_optimization_steps", lower=1, default_value=10, upper=10))
        cs.add_hyperparameter(CS.CategoricalHyperparameter(
            "baseline_optimizer_type", ["adam", "rmsprop"]))
        return cs


class ClassicControlReduced(ClassicControlBase):

    @staticmethod
    def get_configuration_space():
        cs = CS.ConfigurationSpace()

        cs.add_hyperparameter(CS.UniformIntegerHyperparameter(
            "n_units_1", lower=8, default_value=64, upper=128, log=True))
        cs.add_hyperparameter(CS.UniformIntegerHyperparameter(
            "n_units_2", lower=8, default_value=64, upper=128, log=True))
        cs.add_hyperparameter(CS.UniformIntegerHyperparameter(
            "batch_size", lower=8, default_value=64, upper=256, log=True))
        cs.add_hyperparameter(CS.UniformFloatHyperparameter(
            "learning_rate", lower=1e-7, default_value=1e-3, upper=1e-1, log=True))
        cs.add_hyperparameter(CS.UniformFloatHyperparameter(
            "discount", lower=0, default_value=.99, upper=1))
        cs.add_hyperparameter(CS.UniformFloatHyperparameter(
            "likelihood_ratio_clipping", lower=1e-3, default_value=.2, upper=1))
        cs.add_hyperparameter(CS.UniformFloatHyperparameter(
            "entropy_regularization", lower=0, default_value=0.01, upper=1))
        return cs
