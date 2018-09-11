import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
import openml
import ConfigSpace

from hpolib.abstract_benchmark import AbstractBenchmark
from hpolib.util.preprocessing import DataDetergent


class DatasetWrapper(Dataset):

    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        return torch.Tensor(self.data[item]), torch.LongTensor([self.targets[item]])


class Architecture(torch.nn.Module):

    def __init__(self, n_input, n_units_1, n_units_2, dropout_1, dropout_2, n_classes):
        super(Architecture, self).__init__()

        self.fc_1 = torch.nn.Linear(n_input, n_units_1)
        self.fc_2 = torch.nn.Linear(n_units_1, n_units_2)
        self.fc_3 = torch.nn.Linear(n_units_2, n_classes)

        self.dropout_1 = torch.nn.Dropout(dropout_1)
        self.dropout_2 = torch.nn.Dropout(dropout_2)

    def forward(self, input):
        x = F.relu(self.fc_1(input))
        x = self.dropout_1(x)
        x = F.relu(self.fc_2(x))
        x = self.dropout_2(x)
        x = self.fc_3(x)
        return x


class ClassificationNeuralNetwork(AbstractBenchmark):

    def __init__(self, task_id):
        super().__init__()
        task = openml.tasks.get_task(task_id)

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
        self.train_targets = y[train_indices]

        self.train = preproc.fit_transform(X_train)

        X_valid = X[valid_indicies]
        self.valid_targets = y[valid_indicies]

        self.valid = preproc.transform(X_valid)

        self.n_input = self.train.shape[1]
        self.n_classes = np.unique(self.train_targets).shape[0]

    def objective_function(self, configuration, num_epochs=30, **kwargs):

        start_time = time.time()

        net = Architecture(self.n_input, configuration["n_units_1"], configuration["n_units_2"],
                           configuration["dropout_1"], configuration["dropout_2"], self.n_classes)

        data = DatasetWrapper(self.train, self.train_targets)
        trainloader = DataLoader(data, batch_size=configuration["batch_size"])

        data = DatasetWrapper(self.valid, self.valid_targets)
        validloader = DataLoader(data, batch_size=configuration["batch_size"])
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=configuration["lr"])

        learning_curve_valid = []
        runtime = []

        for epoch in range(num_epochs):

            net.train()

            for i, batch in enumerate(trainloader):

                inputs, labels = batch

                optimizer.zero_grad()

                outputs = net(inputs)
                loss = criterion(outputs, labels[:, 0])
                loss.backward()

                optimizer.step()

            total = 0
            correct = 0
            net.eval()

            for i, batch in enumerate(validloader):
                inputs, labels = batch

                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels[:, 0]).sum().item()
            learning_curve_valid.append(1 - correct / total)

            runtime.append(time.time() - start_time)
        return {'function_value': learning_curve_valid[-1], "cost": runtime[-1],
                "lc_valid": learning_curve_valid, "runtime": runtime}

    def objective_function_test(self, configuration, **kwargs):
        return self.objective_function(configuration, num_epochs=30)

    @staticmethod
    def get_configuration_space():

        cs = ConfigSpace.ConfigurationSpace()

        cs.add_hyperparameter(ConfigSpace.UniformFloatHyperparameter(
            'lr', lower=1e-6, upper=1e-1, log=True))

        cs.add_hyperparameter(ConfigSpace.UniformIntegerHyperparameter(
            'batch_size', lower=8, upper=128, log=True))

        cs.add_hyperparameter(ConfigSpace.UniformIntegerHyperparameter(
            'n_units_1', lower=16, upper=512, log=True))

        cs.add_hyperparameter(ConfigSpace.UniformIntegerHyperparameter(
            'n_units_2', lower=16, upper=512, log=True))

        cs.add_hyperparameter(ConfigSpace.UniformFloatHyperparameter(
            'dropout_1', lower=0, upper=.99))

        cs.add_hyperparameter(ConfigSpace.UniformFloatHyperparameter(
            'dropout_2', lower=0, upper=.99))

        return cs

    @staticmethod
    def get_meta_information():
        return {'name': 'neural network for classification task on OpenML datasets',
                'references': []
                }


if __name__ == '__main__':
    b = ClassificationNeuralNetwork(task_id=3)

    cs = b.get_configuration_space()
    config = cs.sample_configuration()
    results = b.objective_function(config)

    print(results)
