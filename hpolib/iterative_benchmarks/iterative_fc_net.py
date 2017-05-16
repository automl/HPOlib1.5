import time
import keras
import numpy as np
import ConfigSpace as CS

from hpolib.abstract_iterative_model import AbstractIterativeModel


class IterativeFCNet(AbstractIterativeModel):

    def __init__(self, config, train, train_targets, valid, valid_targets, test, test_targets, max_steps=100, rng=None):
        """
        Initializes Fully connected network
        Parameters
        ----------

        max_num_epochs: int
            set maximum number of epochs. Needed to calculate how many number
            of epochs to use for training given number of steps in [0, 100].

        rng: str
            set up rng
        """

        self.train = train
        self.train_targets = train_targets
        self.valid = valid
        self.valid_targets = valid_targets
        self.test = test
        self.test_targets = test_targets

        num_classes = self.train_targets.shape[1]

        if rng is None:
            self.rng = np.random.RandomState()
        else:
            self.rng = rng

        super(IterativeFCNet, self).__init__(config, max_steps)

        self.model = keras.models.Sequential()
        self.model.add(keras.layers.Dense(self.config["num_units_1"],
                                          activation='relu',
                                          input_shape=(self.train.shape[1],)))
        self.model.add(keras.layers.Dropout(self.config["dropout_1"]))
        self.model.add(keras.layers.Dense(self.config["num_units_2"], activation='relu'))
        self.model.add(keras.layers.Dropout(self.config["dropout_2"]))
        self.model.add(keras.layers.Dense(num_classes, activation='softmax'))

        #self.model.summary()

        self.model.compile(loss='categorical_crossentropy',
                           optimizer=keras.optimizers.RMSprop(lr=self.config["lr"]),
                           metrics=['accuracy'])

        self.learning_curve = []

    def get_data(self):
        pass

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

        start_time = time.time()
        history = self.model.fit(self.train, self.train_targets,
                                 batch_size=self.config["batch_size"],
                                 epochs=n_steps,
                                 verbose=0,
                                 validation_data=(self.valid, self.valid_targets))
        self.current_step += n_steps

        print(self.current_step)
        self.learning_curve.extend(history.history["val_acc"])

        return {'function_value': 1 - history.history["val_acc"][-1],
                "cost": time.time() - start_time,
                "valid_acc": [1 - history.history["val_acc"][i] for i in range(n_steps)],
                "train_acc": [1 - history.history["acc"][i] for i in range(n_steps)],
                "train_loss": history.history["loss"],
                "valid_loss": history.history["val_loss"]}

    @staticmethod
    def get_configuration_space():
        cs = CS.ConfigurationSpace()

        num_units_1 = CS.hyperparameters.UniformIntegerHyperparameter("num_units_1", 16, 1024, default=64, log=True)
        num_units_2 = CS.hyperparameters.UniformIntegerHyperparameter("num_units_2", 16, 1024, default=64, log=True)
        dropout_1 = CS.hyperparameters.UniformFloatHyperparameter("dropout_1", 0, .99, default=.5)
        dropout_2 = CS.hyperparameters.UniformFloatHyperparameter("dropout_2", 0, .99, default=.5)
        lr = CS.hyperparameters.UniformFloatHyperparameter("lr", 10e-5, 10e-1, default=10e-2, log=True)
        batch_size = CS.hyperparameters.UniformIntegerHyperparameter("batch_size", 8, 512, default=16, log=True)

        cs.add_hyperparameters([num_units_1, num_units_2, dropout_1, dropout_2, lr, batch_size])
        return cs
