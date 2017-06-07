import unittest

import hpolib.util.openml_data_manager


class TestDataManager(unittest.TestCase):
    def test_load_openml(self):
        # Load Data
        dm = hpolib.util.openml_data_manager.OpenMLHoldoutDataManager(openml_task_id=233)
        X_train, y_train, X_val, y_val, X_test, y_test = dm.load()

        # Assert array shape
        n = 3196
        n_test = int(n * 0.33)
        n_train_valid = n - n_test

        n_train = int(n_train_valid * (1 - 0.33))
        n_valid = n_train_valid - n_train
        self.assertEqual(X_train.shape[0], n_train)
        self.assertEqual(X_train.shape[1], 36)
        self.assertEqual(X_val.shape[0], n_valid)
        self.assertEqual(X_val.shape[1], 36)
        self.assertEqual(X_test.shape[0], n_test)
        self.assertEqual(X_test.shape[1], 36)

        self.assertEqual(y_train.shape[0], n_train)
        self.assertEqual(y_val.shape[0], n_valid)
        self.assertEqual(y_test.shape[0], n_test)