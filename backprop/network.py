"""
network.py
"""

import random
import numpy as np
import math
from tqdm import tqdm
import matplotlib.pyplot as plt


class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.current_db = np
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def SGD(self, training_data, epochs, mini_batch_size, learning_rate,
            test_data):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  """
        print("Initial test accuracy: {0}".format(self.one_label_accuracy(test_data)))
        n = len(training_data)
        # train_acc = []
        # train_loss = []
        # test_acc = []
        euc_norms = []
        for j in tqdm(range(epochs)):
            random.shuffle(list(training_data))
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate)
            db_norms = []
            for i, dbi in enumerate(self.current_db):
                norm = np.linalg.norm(dbi)
                norm = norm / float(len(training_data))
                db_norms.append(norm)
            euc_norms.append(db_norms)
        # for xe, ye in zip(range(epochs), euc_norms)
        #     plt.scatter([xe]*len(ye), ye)
        # plt.xticks(range(epochs))
        plt.plot(range(epochs), euc_norms)
        plt.xlabel('Epoch')
        plt.ylabel('Euclidean Norms')
        plt.savefig('deucnorms.jpg')
        plt.show()


            # train_acc.append(self.one_hot_accuracy(training_data))
            # train_loss.append(self.loss(training_data))
            # test_acc.append(self.one_label_accuracy(test_data))

            #print ("Epoch {0} test accuracy: {1}".format(j, self.one_label_accuracy(test_data)))
        print("Epoch Final test accuracy: {0}".format(self.one_label_accuracy(test_data)))
        # tacc, = plt.plot(range(epochs), train_acc, label='Train Accuracy', marker=(1,0))
        # plt.legend(handles=[tacc])
        # plt.xlabel('Epoch')
        # plt.ylabel('Accuracy')
        # plt.savefig('trainacc.jpg')
        # plt.show()
        # tlss, = plt.plot(range(epochs), train_loss, label='Train Loss', marker=(2,0))
        # plt.legend(handles=[tlss])
        # plt.xlabel('Epoch')
        # plt.ylabel('Loss')
        # plt.savefig('trainloss.jpg')
        # plt.show()
        # tstacc, = plt.plot(range(epochs), test_acc, label='Test Accuracy', marker=(3,0))
        # plt.legend(handles=[tstacc])
        # plt.xlabel('Epoch')
        # plt.ylabel('Accuracy')
        # plt.savefig('testacc_c.jpg')
        # plt.show()



    def update_mini_batch(self, mini_batch, learning_rate):
        """Update the network's weights and biases by applying
        stochastic gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (learning_rate / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (learning_rate / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """ Given x,y feed forward, calc loss and backprop through the network
            Returning db, dw - the gradients of the loss w.r to each b and w
        """
        #feed forward
        pre_activations = []  # z 1-L
        activations = []  # a 0 - L
        #pre_activations.append(x)
        activations.append(x)
        layer = 0
        for w, b in zip(self.weights, self.biases):
            x = w.dot(x) + b # linear
            pre_activations.append(x)
            if not (layer == len(self.weights) - 1):
                x = sigmoid(x)
            activations.append(x)
            layer += 1

        deltas = []
        delta = self.loss_derivative_wr_output_activations(activations[-1], y) # delta_L
        deltas.append(delta)

        for i, preact in enumerate(pre_activations[::-1]):
            if not i == 0:
                delta_l1 = deltas[i-1]
                weight = self.weights[-i]
                sigma_tag = sigmoid_derivative(preact)
                delta_l = (weight.transpose().dot(delta_l1)) * sigma_tag
                deltas.append(delta_l)
        
        dw = []
        db = deltas[::-1]

        for i, delta in enumerate(deltas[::-1]):
            active = activations[i]
            cdw = delta.dot(active.transpose())
            dw.append(cdw)
        self.current_db = db
        return db, dw

    def one_label_accuracy(self, data):
        """Return accuracy of network on data with numeric labels"""
        output_results = [(np.argmax(self.network_output_before_softmax(x)), y)
         for (x, y) in data]
        return sum(int(x == y) for (x, y) in output_results)/float(len(data))

    def one_hot_accuracy(self,data):
        """Return accuracy of network on data with one-hot labels"""
        output_results = [(np.argmax(self.network_output_before_softmax(x)), np.argmax(y))
                          for (x, y) in data]
        return sum(int(x == y) for (x, y) in output_results) / float(len(data))


    def network_output_before_softmax(self, x):
        """Return the output of the network before softmax if ``x`` is input."""
        layer = 0
        for b, w in zip(self.biases, self.weights):
            if layer == len(self.weights) - 1:
                x = np.dot(w, x) + b
            else:
                x = sigmoid(np.dot(w, x)+b)
            layer += 1
        return x

    def loss(self, data):
        """Return the loss of the network on the data"""
        loss_list = []
        for (x, y) in data:
            net_output_before_softmax = self.network_output_before_softmax(x)
            net_output_after_softmax = self.output_softmax(net_output_before_softmax)
            loss_list.append(np.dot(-np.log(net_output_after_softmax).transpose(),y).flatten()[0])
        return sum(loss_list) / float(len(data))

    def output_softmax(self, output_activations):
        """Return output after softmax given output before softmax"""
        output_exp = np.exp(output_activations)
        return output_exp/output_exp.sum()

    def loss_derivative_wr_output_activations(self, output_activations, y):
        """Return derivative of loss with respect to the output activations before softmax"""
        return self.output_softmax(output_activations) - y


def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_derivative(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
