import openml

from hpolib.util.preprocessing import DataDetergent
from hpolib.benchmarks.ml.svm_benchmark import SupportVectorMachine


class SupportVectorMachineOpenML100(SupportVectorMachine):

    def __init__(self, task_id, rng=None):
        """

        Parameters
        ----------
        rng: int/None/RandomState
            set up rng
        """

        self.task_id = task_id
        super().__init__(rng)

    def get_data(self):
        task = openml.tasks.get_task(self.task_id)

        dataset = task.get_dataset()
        _, _, categorical_indicator = dataset.get_data(
            target=task.target_name,
            return_categorical_indicator=True)

        cat_variables = [True if ci else False
                         for ci in categorical_indicator]

        preproc = DataDetergent(one_hot_encoding=True, remove_constant_features=True, normalize=True,
                                categorical_features=cat_variables, impute_nans=True)

        train_indices, valid_indicies = task.get_train_test_split_indices()

        X, y = task.get_X_and_y()
        X_train = X[train_indices]
        train_targets = y[train_indices]

        train = preproc.fit_transform(X_train)

        X_valid = X[valid_indicies]
        valid_targets = y[valid_indicies]

        valid = preproc.transform(X_valid)

        return train, train_targets, valid, valid_targets, None, None
