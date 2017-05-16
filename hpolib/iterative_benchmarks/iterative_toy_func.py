import numpy as np
import ConfigSpace as CS

from hpolib.abstract_iterative_model import AbstractIterativeModel
from hpolib.abstract_iterative_benchmark import AbstractIterativeBenchmark


class IterativeToyFunc(AbstractIterativeModel):

    def __init__(self, config, max_steps=81, rng=None):
        super(IterativeToyFunc, self).__init__(config, max_steps)
        self.learning_curve = []

    def run(self, n_steps, **kwargs):

        #noise = np.random.randn() * 10e-4
        noise = 0
        y = [1 - (10 + self.config["a"] * np.log(self.config["b"] * self.current_step + i + 10e-4)) / 10. + noise
             for i in range(n_steps)]

        self.current_step += n_steps

        self.learning_curve.extend(y)
        return {'function_value': y[-1]}

    @staticmethod
    def get_configuration_space():
        cs = CS.ConfigurationSpace()

        a = CS.hyperparameters.UniformFloatHyperparameter("a", 0, 1, default=.5)
        b = CS.hyperparameters.UniformFloatHyperparameter("b", 0, 1, default=.5)

        cs.add_hyperparameters([a, b])
        return cs

    def run_test(self, n_steps, **kwargs):

        return self.run()

    @staticmethod
    def get_meta_information():
        return {'name': 'Toy function'}


class IterativeToyFuncFactory(AbstractIterativeBenchmark):

    def __init__(self, max_steps):
        self.max_steps = max_steps

    def get_data(self):
        pass

    def get_model(self, config=None):

        if config is None:
            config = IterativeToyFunc.get_configuration_space().sample_configuration()

        return IterativeToyFunc(config=config, max_steps=self.max_steps)

    @staticmethod
    def get_configuration_space():
        return IterativeToyFunc.get_configuration_space()
