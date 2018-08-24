import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from torchvision import datasets, transforms


import ConfigSpace as CS

import hpolib
from hpolib.abstract_benchmark import AbstractBenchmark
from hpolib.util.data_manager import HoldoutDataManager
import hpolib.util.rng_helper as rng_helper


class ConvNetworkPytorch(AbstractBenchmark):

    def __init__(self, max_epochs=10, batch_size=64, rng=None):
        """
        ConvNet (based on the Pytorch tutorial:
                 https://github.com/pytorch/examples/blob/master/mnist/main.py)


        Parameters
        ----------
        rng: str
            set up rng
        """
        super(ConvNetworkPytorch, self).__init__(rng=rng)

        self.train_data, self.valid_data, self.test_data = self.get_data()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.max_epochs = max_epochs

    def get_data(self):
        raise NotImplementedError("Do not use this benchmark as this is only "
                                  "a skeleton for further implementations.")

    @AbstractBenchmark._check_configuration
    @AbstractBenchmark._configuration_as_array
    def objective_function(self, x, **kwargs):
        init_lr = 10**x[0]
        weight_decay = 10 ** x[1]
        momentum = x[2]
        dropout_rate = x[3]

        rng = kwargs.get("rng", None)
        self.rng = rng_helper.get_rng(rng=rng, self_rng=self.rng)
        if rng is not None:
            torch.manual_seed(self.rng)

        train_loader = torch.utils.data.DataLoader(self.train_data,
                                                   batch_size=self.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(self.valid_data,
                                                  batch_size=self.batch_size*10, shuffle=True)

        learning_curve, cost_curve, train_loss, valid_loss = self._train_net(init_lr=init_lr,
                                                                             momentum=momentum,
                                                                             weight_decay=weight_decay,
                                                                             dropout_rate=dropout_rate,
                                                                             train_loader=train_loader,
                                                                             valid_loader=test_loader)
        y = learning_curve[-1]
        c = cost_curve[-1]
        return {'function_value': y,
                "cost": c,
                "learning_curve": learning_curve,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "learning_curve_cost": cost_curve}

    @AbstractBenchmark._check_configuration
    @AbstractBenchmark._configuration_as_array
    def objective_function_test(self, x, **kwargs):
        init_lr = 10**x[0]
        weight_decay = 10 ** x[1]
        momentum = x[2]
        dropout_rate = x[3]

        rng = kwargs.get("rng", None)
        self.rng = rng_helper.get_rng(rng=rng, self_rng=self.rng)
        if rng is not None:
            torch.manual_seed(self.rng)

        train_loader = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset([self.train_data, self.valid_data]),
                                                   batch_size=self.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(self.test_data,
                                                  batch_size=self.batch_size*10, shuffle=True)

        learning_curve, cost_curve, train_loss, valid_loss = self._train_net(init_lr=init_lr,
                                                                             momentum=momentum,
                                                                             weight_decay=weight_decay,
                                                                             dropout_rate=dropout_rate,
                                                                             train_loader=train_loader,
                                                                             valid_loader=test_loader)
        y = learning_curve[-1]
        c = cost_curve[-1]
        return {'function_value': y, "cost": c, "learning_curve": learning_curve}

    @staticmethod
    def get_configuration_space():
        cs = CS.ConfigurationSpace(seed=np.random.randint(1, 100000))
        cs.generate_all_continuous_from_bounds(ConvNetworkPytorch.
                                               get_meta_information()['bounds'])
        return cs

    @staticmethod
    def get_meta_information():
        return {'name': 'Convolutional Connected Network Pytorch',
                'bounds': [[-5.0, -1.0],  # init_learning_rate
                           [-6.0, -2.0],  # l2_reg
                           [0.3, 1-1e-5], # momentum
                           [0.1, 0.9]     # dropout_rate
                           ],
                'num_function_evals': 50,
                'note': "Uses as many cores as are available. Can be fixed with OMP_NUM_THREADS=1",
                'requires': ["pytorch",
                             "torchvision"],
                'references': ["None"]
                }

    def __train(self, model, train_loader, optimizer):
        # train model for one epoch
        model.train()
        train_acc = 0
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            train_acc += pred.eq(target.view_as(pred)).sum().item()
        train_acc = 100. * train_acc / len(train_loader.dataset)
        train_loss /= len(train_loader.dataset)
        return train_acc, train_loss

    def __test(self, model, test_loader):
        # get validation accuracy for test_loader
        # returns loss and accuracy
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        test_acc = 100. * correct / len(test_loader.dataset)
        return test_loss, test_acc

    def _train_net(self, init_lr, momentum, weight_decay, dropout_rate, train_loader, valid_loader):
        start_time = time.time()
        model = _ConvNet(dropout_rate=dropout_rate).to(self.device)

        optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=momentum, weight_decay=weight_decay)

        learning_curve = []
        cost_curve = []
        train_loss = []
        valid_loss = []
        for epoch in range(1, self.max_epochs + 1):
            _, trn_loss = self.__train(model=model, train_loader=train_loader, optimizer=optimizer)
            val_acc, val_loss = self.__test(model=model, test_loader=valid_loader)

            learning_curve.append(1 - val_acc)
            train_loss.append(trn_loss)
            valid_loss.append(val_loss)
            cost_curve.append(time.time() - start_time)

        return learning_curve, cost_curve, train_loss, valid_loss


class _ConvNet(nn.Module):
    def __init__(self, dropout_rate):
        super(_ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(p=dropout_rate)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class MNISTPytorchData(HoldoutDataManager):

    def __init__(self):
        # Loads data as
        self.save_to = os.path.join(hpolib._config.data_dir, "MNISTPytorch")
        super(MNISTPytorchData, self).__init__()

    def load(self):
        trn_d = datasets.MNIST(self.save_to, train=True, download=True,
                                 transform=transforms.Compose([transforms.ToTensor(),
                                                               transforms.Normalize((0.1307,), (0.3081,))]))
        trn_d, val_d = torch.utils.data.random_split(trn_d, [50000, 10000])
        tst_d = datasets.MNIST(self.save_to, train=False, download=True,
                               transform=transforms.Compose([transforms.ToTensor(),
                                                             transforms.Normalize((0.1307,), (0.3081,))]))
        return trn_d, val_d, tst_d


class ConvNetworkPytorchOnMnist(ConvNetworkPytorch):

    def get_data(self):
        dm = MNISTPytorchData()
        return dm.load()

    @staticmethod
    def get_meta_information():
        d = ConvNetworkPytorch.get_meta_information()
        data_ref = ["@article{lecun-ieee98,"
                    "title={Gradient-based learning applied to document recognition},"
                    "author={Y. LeCun and L. Bottou and Y. Bengio and P. Haffner},"
                    "journal={Proceedings of the IEEE},"
                    "pages={2278--2324},"
                    "year={1998},"
                    "publisher={IEEE}"]
        d["references"].append(data_ref)
        return d
