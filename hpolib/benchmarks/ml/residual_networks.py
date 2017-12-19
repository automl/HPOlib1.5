import time
import numpy as np
import lasagne
import theano
import theano.tensor as T

import ConfigSpace as CS

from hpolib.abstract_benchmark import AbstractBenchmark
from hpolib.util.data_manager import CIFAR10Data


class ResidualNeuralNetwork(AbstractBenchmark):

    def __init__(self, max_num_epochs=40, rng=None):
        self.X_train, self.y_train, self.X_valid, self.y_valid, self.X_test, self.y_test = self.get_data()

        # Augment data by flipping the images
        X_train_flip = self.X_train[:, :, :, ::-1]
        y_train_flip = self.y_train

        self.X_train = np.concatenate((self.X_train, X_train_flip), axis=0)
        self.y_train = np.concatenate((self.y_train, y_train_flip), axis=0)

        self.X_train = np.array(self.X_train, dtype=np.float32)
        self.y_train = np.array(self.y_train, dtype=np.int32)
        self.X_valid = np.array(self.X_valid, dtype=np.float32)
        self.y_valid = np.array(self.y_valid, dtype=np.int32)
        self.X_test = np.array(self.X_test, dtype=np.float32)
        self.y_test = np.array(self.y_test, dtype=np.int32)

        self.max_num_epochs = max_num_epochs
        self.num_epochs = 82
        self.s_min = 256 / self.X_train.shape[0]
        self.s_max = self.X_train.shape[0]

        self.num_classes = len(np.unique(self.y_train))

        if rng is None:
            self.rng = np.random.RandomState()
        else:
            self.rng = rng

        self.input_var = T.tensor4('inputs')
        self.target_var = T.ivector('targets')

        lasagne.random.set_rng(self.rng)
        super(ResidualNeuralNetwork, self).__init__()

    def get_data(self, path):
        pass

    @AbstractBenchmark._check_configuration
    @AbstractBenchmark._configuration_as_array
    def objective_function(self, x, steps=1, dataset_fraction=1, **kwargs):

        size = int(self.X_train.shape[0] * dataset_fraction)

        lr = np.float32(np.power(10, x[0]))
        l2_fac = np.float32(np.power(10, x[1]))
        lr_factor = np.float32(np.power(10, x[2]))
        momentum = np.float32(x[3])

        start_time = time.time()

        # Create neural network model
        print("Building model and compiling functions...")
        self.network = self.build_cnn(self.input_var)
        print("number of parameters in model: %d" % lasagne.layers.count_params(self.network, trainable=True))

        shuffle = np.random.permutation(np.arange(self.X_train.shape[0]))

        X_train = self.X_train[shuffle[:size]]
        y_train = self.y_train[shuffle[:size]]

        valid_error = self.run_training(lr, l2_fac, lr_factor, momentum, X_train, y_train, self.X_valid, self.y_valid)

        cost = time.time() - start_time

        return {'function_value': valid_error,
                "cost": cost}

    @AbstractBenchmark._check_configuration
    @AbstractBenchmark._configuration_as_array
    def objective_function_test(self, x, steps=1, **kwargs):

        start_time = time.time()

        X = np.concatenate((self.X_train, self.X_valid), axis=0)
        y = np.concatenate((self.y_train, self.y_valid), axis=0)

        lr = np.float32(np.power(10, x[0]))
        l2_fac = np.float32(np.power(10, x[1]))
        lr_factor = np.float32(np.power(10, x[2]))
        momentum = np.float32(x[3])

        self.network = self.build_cnn(self.input_var)

        test_error = self.run_training(lr, l2_fac, lr_factor, momentum, X, y, self.X_test, self.y_test)

        cost = time.time() - start_time

        return {'function_value': test_error,
                "cost": cost}

    @staticmethod
    def get_configuration_space():
        cs = CS.ConfigurationSpace()
        cs.generate_all_continuous_from_bounds(ResidualNeuralNetwork.get_meta_information()['bounds'])
        return cs

    def build_cnn(self, input_var=None, n=5):

        # create a residual learning building block with two stacked 3x3
        # convlayers as in paper
        def residual_block(l, increase_dim=False, projection=False):
            input_num_filters = l.output_shape[1]
            if increase_dim:
                first_stride = (2, 2)
                out_num_filters = input_num_filters * 2
            else:
                first_stride = (1, 1)
                out_num_filters = input_num_filters

            stack_1 = lasagne.layers.normalization.batch_norm(
                lasagne.layers.Conv2DLayer(l, num_filters=out_num_filters,
                                       filter_size=(3, 3),
                                       stride=first_stride,
                                       nonlinearity=lasagne.nonlinearities.rectify, pad='same',
                                       W=lasagne.init.HeNormal(gain='relu')))
            stack_2 = lasagne.layers.normalization.batch_norm(lasagne.layers.Conv2DLayer(stack_1,
                                       num_filters=out_num_filters,
                                       filter_size=(3, 3),
                                       stride=(1, 1),
                                       nonlinearity=None,
                                       pad='same',
                                       W=lasagne.init.HeNormal(gain='relu')))

            # add shortcut connections
            if increase_dim:
                if projection:
                    # projection shortcut, as option B in paper
                    projection = lasagne.layers.normalization.batch_norm(lasagne.layers.Conv2DLayer(l,
                                        num_filters=out_num_filters,
                                        filter_size=(1, 1),
                                        stride=(2, 2),
                                        nonlinearity=None, pad='same', b=None))
                    block = lasagne.layers.NonlinearityLayer(lasagne.layers.ElemwiseSumLayer([stack_2, projection]),
                                              nonlinearity=lasagne.nonlinearities.rectify)
                else:
                    # identity shortcut, as option A in paper
                    identity = lasagne.layers.ExpressionLayer(l, lambda X: X[:, :, ::2, ::2],
                                               lambda s: (s[0], s[1], s[2] // 2, s[3]//2))
                    padding = lasagne.layers.PadLayer(identity, [out_num_filters // 4, 0, 0],
                                       batch_ndim=1)
                    block = lasagne.layers.NonlinearityLayer(lasagne.layers.ElemwiseSumLayer([stack_2, padding]),
                                                             nonlinearity=lasagne.nonlinearities.rectify)
            else:
                block = lasagne.layers.NonlinearityLayer(lasagne.layers.ElemwiseSumLayer([stack_2, l]),
                                          nonlinearity=lasagne.nonlinearities.rectify)

            return block

        # Building the network
        l_in = lasagne.layers.InputLayer(shape=(None, 3, 32, 32), input_var=input_var)

        # first layer, output is 16 x 32 x 32
        l = lasagne.layers.normalization.batch_norm(lasagne.layers.Conv2DLayer(l_in, num_filters=16,
                                 filter_size=(3, 3),
                                 stride=(1, 1),
                                 nonlinearity=lasagne.nonlinearities.rectify, pad='same',
                                 W=lasagne.init.HeNormal(gain='relu')))

        # first stack of residual blocks, output is 16 x 32 x 32
        for _ in range(n):
            l = residual_block(l)

        # second stack of residual blocks, output is 32 x 16 x 16
        l = residual_block(l, increase_dim=True)
        for _ in range(1, n):
            l = residual_block(l)

        # third stack of residual blocks, output is 64 x 8 x 8
        l = residual_block(l, increase_dim=True)
        for _ in range(1, n):
            l = residual_block(l)

        # average pooling
        l = lasagne.layers.GlobalPoolLayer(l)

        # fully connected layer
        network = lasagne.layers.DenseLayer(
                l, num_units=10,
                W=lasagne.init.HeNormal(),
                nonlinearity=lasagne.nonlinearities.softmax)

        return network

    def iterate_minibatches(self, inputs, targets, batchsize,
                            shuffle=False, augment=False):
        assert len(inputs) == len(targets)
        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            if augment:
                # as in paper :
                # pad feature arrays with 4 pixels on each side
                # and do random cropping of 32x32
                padded = np.pad(inputs[excerpt], ((0, 0), (0, 0), (4, 4), (4, 4)),
                                mode='constant')
                random_cropped = np.zeros(inputs[excerpt].shape, dtype=np.float32)
                crops = np.random.random_integers(0, high=8, size=(batchsize,2))
                for r in range(batchsize):
                    random_cropped[r, :, :, :] = padded[r, :, crops[r,0]:(crops[r,0] + 32), crops[r,1]:(crops[r,1]+32)]
                inp_exc = random_cropped
            else:
                inp_exc = inputs[excerpt]

            yield inp_exc, targets[excerpt]

    def run_training(self, lr, l2_fac, lr_factor, momentum,
              X_train, Y_train, X_valid, Y_valid):

        prediction = lasagne.layers.get_output(self.network)
        loss = lasagne.objectives.categorical_crossentropy(prediction,
                                                           self.target_var)
        loss = loss.mean()
        # add weight decay
        all_layers = lasagne.layers.get_all_layers(self.network)
        l2_penalty = lasagne.regularization.regularize_layer_params(all_layers,
                                        lasagne.regularization.l2) * l2_fac
        loss = loss + l2_penalty

        params = lasagne.layers.get_all_params(self.network, trainable=True)

        sh_lr = theano.shared(lasagne.utils.floatX(lr))
        updates = lasagne.updates.momentum(
                loss, params, learning_rate=sh_lr, momentum=momentum)

        train_fn = theano.function([self.input_var, self.target_var],
                                   loss, updates=updates)

        test_prediction = lasagne.layers.get_output(self.network,
                                                deterministic=True)
        test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                                self.target_var)
        test_loss = test_loss.mean()
        test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1),
                               self.target_var),
                               dtype=theano.config.floatX)

        val_fn = theano.function([self.input_var, self.target_var], [test_loss, test_acc])

        # launch the training loop
        print("Starting training...")

        # We iterate over epochs:
        for epoch in range(self.num_epochs):
            # shuffle training data
            train_indices = np.arange(X_train.shape[0])
            np.random.shuffle(train_indices)
            X_train = X_train[train_indices, :, :, :]
            Y_train = Y_train[train_indices]

            # In each epoch, we do a full pass over the training data:
            train_err = 0
            train_batches = 0
            start_time = time.time()
            for batch in self.iterate_minibatches(X_train, Y_train, 128,
                                            shuffle=True, augment=True):
                inputs, targets = batch
                train_err += train_fn(inputs, targets)
                train_batches += 1

            # And a full pass over the validation data:
            val_err = 0
            val_acc = 0
            val_batches = 0
            for batch in self.iterate_minibatches(X_valid, Y_valid, 500,
                                                  shuffle=False):
                inputs, targets = batch
                err, acc = val_fn(inputs, targets)
                val_err += err
                val_acc += acc
                val_batches += 1

            # Then we print the results for this epoch:
            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, self.num_epochs, time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
            print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
            print("  validation accuracy:\t\t{:.2f} %".format(
                val_acc / val_batches * 100))

            # adjust learning rate as in paper
            # 32k and 48k iterations should be roughly equivalent to 41 and 61 epochs
            if (epoch + 1) == 41 or (epoch + 1) == 61:
                new_lr = sh_lr.get_value() * lr_factor
                print("New LR:" + str(new_lr))
                sh_lr.set_value(lasagne.utils.floatX(new_lr))

        # Calculate validation error of model:
        test_err = 0
        test_acc = 0
        test_batches = 0
        for batch in self.iterate_minibatches(X_valid, Y_valid, 500, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            test_err += err
            test_acc += acc
            test_batches += 1
        print("Final results:")
        print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
        print("  test accuracy:\t\t{:.2f} %".format(
            test_acc / test_batches * 100))

        acc = test_acc / test_batches * 100
        return 1 - acc / 100.

    @staticmethod
    def get_meta_information():
        return {'name': 'Residual Neural Network',
                'bounds': [[-6, 0],  # Learning rate
                           [-6, 0],  # L2 factor
                           [-4, 0],  # Learning rate factor
                           [0.1, 0.999]],  # Momentum
                'references': []
                }


class ResidualNeuralNetworkOnCIFAR10(ResidualNeuralNetwork):

    def get_data(self):
        dm = CIFAR10Data()
        return dm.load()
