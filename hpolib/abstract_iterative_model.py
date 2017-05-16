import abc

import ConfigSpace


class AbstractIterativeModel(object, metaclass=abc.ABCMeta):

    def __init__(self, config, max_steps):
        """Interface for benchmarks.

        A benchmark contains of two building blocks, the target function and
        the configuration space. Furthermore it can contain additional
        benchmark-specific information such as the location and the function
        value of the global optima. New benchmarks should be derived from
        this base class or one of its child classes.
        """
        self.config = config
        self.max_steps = max_steps
        self.current_step = 0
        self.configuration_space = self.get_configuration_space()

    @abc.abstractmethod
    def run(self, n_steps, **kwargs):
        """Objective function.

        Override this function to provide your benchmark function. This
        function will be called by one of the evaluate functions. For
        flexibility you have to return a dictionary with the only mandatory
        key being `function_value`, the objective function value for the
        configuration which was passed. By convention, all benchmarks are
        minimization problems.

        Parameters
        ----------
        n_steps : int
            Number of steps to run

        Returns
        -------
        dict
            Must contain at least the key `function_value`.
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def get_configuration_space():
        """ Defines the configuration space for each benchmark.

        Returns
        -------
        ConfigSpace.ConfigurationSpace
            A valid configuration space for the benchmark's parameters
        """
        raise NotImplementedError()
