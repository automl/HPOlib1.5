import openml
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split

from hpolib.benchmarks.ml.svm_benchmark import SupportVectorMachine
from hpolib.util.preprocessing import DataDetergent


class SupportVectorMachineOpenML(SupportVectorMachine):

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

        # split in training and test
        train_valid_indices, test_indicies = task.get_train_test_split_indices()

        X, y = task.get_X_and_y()
        if type(X) == csr_matrix:
            X = X.todense()

        X_train_valid = X[train_valid_indices]
        train_valid_targets = y[train_valid_indices]

        X_test = X[test_indicies]
        test_targets = y[test_indicies]

        # split training in training and validation
        X_train, X_valid, train_targets, valid_targets = train_test_split(X_train_valid, train_valid_targets,
                                                                          test_size=0.33)

        # preprocess data
        train = preproc.fit_transform(X_train)
        valid = preproc.transform(X_valid)
        test = preproc.transform(X_test)

        return train, train_targets, valid, valid_targets, test, test_targets
