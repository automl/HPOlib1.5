import abc

import numpy as np

from hpolib.abstract_benchmark import AbstractBenchmark


class AbstractMultitaskBenchmark(AbstractBenchmark):

    @abc.abstractmethod
    def objective_function(self, configuration, task, **kwargs):
        """Objective function.

        Override this function to provide your benchmark function. This
        function will be called by one of the evaluate functions. For
        flexibility you have to return a dictionary with the only mandatory
        key being `function_value`, the objective function value for the
        configuration which was passed. By convention, all benchmarks are
        minimization problems.

        Parameters
        ----------
        configuration : dict-like

        task : object
            Task identifier on which the configuration will be evaluated.

        Returns
        -------
        dict
            Must contain at least the key `function_value`.
        """
        pass

    @abc.abstractmethod
    def objective_function_test(self, configuration, task, **kwargs):
        """
        If there is a different objective function for offline testing, e.g
        testing a machine learning on a hold extra test set instead
        on a validation set override this function here.

        Parameters
        ----------
        configuration : dict-like

        task : object
            Task identifier on which the configuration will be evaluated.

        Returns
        -------
        dict
            Must contain at least the key `function_value`.
        """
        pass

    def test(self, n_runs=5, *args, **kwargs):
        """ Draws some random (configuration, task)-pairs and call
        objective_fucntion(_test).

        Parameters
        ----------
        n_runs : int
            number of random configurations to draw and evaluate
        """
        train_rvals = []
        test_rvals = []

        for _ in range(n_runs):
            configuration = self.configuration_space.sample_configuration()
            task = np.random.choice(list(self.get_task_information()))
            train_rvals.append(
                self.objective_function(
                    configuration, task, *args, **kwargs
                )
            )
            test_rvals.append(
                self.objective_function_test(
                    configuration, task, *args, **kwargs
                )
            )

        return train_rvals, test_rvals

    def get_task_information(self):
        """Return information about available tasks.

        Return
        ------
        dict
        """
        pass
