import time
import numpy as np
import ConfigSpace as CS

from tensorforce.agents import PPOAgent
from tensorforce.execution import Runner
from tensorforce.contrib.openai_gym import OpenAIGym

from hpolib.abstract_benchmark import AbstractBenchmark
from hpolib.util import rng_helper



class CartpoleBase(AbstractBenchmark):
	def __init__(self, rng=None, defaults=None, max_budget=9):
		"""
		Parameters
		----------
		rng: int/None/RandomState
			set up rng
		defaults: dict
			default configuration used for the PPO agent
		"""

		super(CartpoleBase, self).__init__()

		self.rng = rng_helper.create_rng(rng)
		
		self.env = OpenAIGym('CartPole-v0', visualize=False)
		self.max_episodes = 3000
		self.avg_n_episodes = 20
		self.max_budget = max_budget
		
		
		self.defaults = {
			"n_units_1": 					64,
			"n_units_2": 					64,
			"batch_size": 					64,
			"learning_rate": 				1e-3,
			"discount": 					0.99,
			"likelihood_ratio_clipping":	0.2,
			"activation_1": 				"tanh",
			"activation_2": 				"tanh",
			"optimizer_type": 				"adam",
			"optimization_steps": 			10,

			"baseline_mode": 				"states",
			"baseline_n_units_1":			64,
			"baseline_n_units_2":			64,
			"baseline_learning_rate":		1e-3,
			"baseline_optimization_steps":	10,
			"baseline_optimizer_type":		"adam"
		}
		
		if not defaults is None:
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

		converged_episodes = []

		for i in range(budget):
			agent = PPOAgent(
				states=self.env.states,
				actions=self.env.actions,
				network=network_spec,
				
				
				 update_mode=dict(	unit='episodes',
									batch_size=config["batch_size"]),

				# BatchAgent
				#keep_last_timestep=True,
				# PPOAgent

				step_optimizer=dict(
					type=config["optimizer_type"],
					learning_rate=config["learning_rate"]
				),
				optimization_steps=config["optimization_steps"],

				# Model
				discount=config["discount"],
				# PGModel
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
				# PGLRModel
				likelihood_ratio_clipping=config["likelihood_ratio_clipping"],
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
		return(self.objective_function(config, budget=self.max_budget, **kwargs))


	@staticmethod
	def get_meta_information():
		return {'name': 'Cartpole',
				'references': []
				}











class CartpoleFull(CartpoleBase):

	@staticmethod
	def get_configuration_space():
		cs = CS.ConfigurationSpace()

		cs.add_hyperparameter(CS.UniformIntegerHyperparameter(
			"n_units_1",lower=8, default_value=64, upper=64, log=True))
		cs.add_hyperparameter(CS.UniformIntegerHyperparameter(
			"n_units_2", lower=8, default_value=64, upper=64, log=True))
		cs.add_hyperparameter(CS.UniformIntegerHyperparameter(
			"batch_size", lower=8, default_value=64, upper=256, log=True))
		cs.add_hyperparameter(CS.UniformFloatHyperparameter(
			"learning_rate", lower=1e-7, default_value=1e-3, upper=1e-1, log=True))
		cs.add_hyperparameter(CS.UniformFloatHyperparameter(
			"discount", lower=0, default_value=.99, upper=1))
		cs.add_hyperparameter(CS.UniformFloatHyperparameter(
			"likelihood_ratio_clipping", lower=0, default_value=.2, upper=1))
		cs.add_hyperparameter(CS.CategoricalHyperparameter(
			"activation_1", ["tanh", "relu"]))
		cs.add_hyperparameter(CS.CategoricalHyperparameter(
			"activation_2", ["tanh", "relu"]))
		cs.add_hyperparameter(CS.CategoricalHyperparameter(
			"optimizer_type", ["adam", "rmsprop"]))
		cs.add_hyperparameter(CS.UniformIntegerHyperparameter(
			"optimization_steps",  lower=1, default_value=10, upper=10))
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




class CartpoleReduced(CartpoleBase):

	@staticmethod
	def get_configuration_space():
		cs = CS.ConfigurationSpace()

		cs.add_hyperparameter(CS.UniformIntegerHyperparameter(
			"n_units_1", lower=8, default_value=64, upper=128, log=True))
		cs.add_hyperparameter(CS.UniformIntegerHyperparameter(
			"n_units_2", lower=8, default_value=64, upper=128, log=True))
		cs.add_hyperparameter(CS.UniformIntegerHyperparameter(
			"batch_size",lower=8, default_value=64, upper=256, log=True))
		cs.add_hyperparameter(CS.UniformFloatHyperparameter(
			"learning_rate", lower=1e-7, default_value=1e-3, upper=1e-1, log=True))
		cs.add_hyperparameter(CS.UniformFloatHyperparameter(
			"discount", lower=0, default_value=.99, upper=1))
		cs.add_hyperparameter(CS.UniformFloatHyperparameter(
			"likelihood_ratio_clipping", lower=0, default_value=.2, upper=1))
		cs.add_hyperparameter(CS.UniformFloatHyperparameter(
			"entropy_regularization", lower=0, default_value=0.01, upper=1))
		return cs
