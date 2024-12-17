from training_data import trainX, trainY, testX, testY
from activations import *

import pickle

import numpy as np
np.random.seed(1)


class Network:
    def __init__(self, architecture):
        self.architecture = architecture
        
        self.input_size = architecture["input_size"]
        self.output_size = architecture["neurons"][-1]
        
        self.layers = architecture["neurons"]
        self.activations = architecture["activations"]

    def calculate_parameter_count(self):
        parameters = 0
        layers = self.layers.prepend(self.input_size)
        for i in range(len(layers) - 1):
            parameters += layers[i] * self.layers[i+1]
        parameters += sum(layers[1:-1])
        return parametsrs

    def create(self):
        # self.weights and self.biases have the same number of values as self.layers
        # the structure of the model is input layer -> (weights + biases -> activation ->) repeated
        
        self.biases = []
        self.weights = []
    
        for i in range(len(self.layers)):
            if i == 0:
                self.weights.append(np.random.normal(0, 1, (self.input_size, self.layers[i])))
            else:
                self.weights.append(np.random.normal(0, 1, (self.layers[i-1], self.layers[i])))
            self.biases.append(np.random.normal(0, 1, (self.layers[i], 1)))

    def forward_pass(self, input):
        inputs = input
        for i in range(len(self.layers)):
            linear_model = np.dot(inputs, self.weights[i]) + self.biases[i].T
            inputs = activation_function.apply(linear_model, self.activations[i])
        return inputs

    def backward_pass(self):
        pass

    def save_parameters(self):
        with open("model.pkl", "wb") as f:
            pickle.dump(self, f)


architecture = {
    "input_size": 784,
    "neurons": [20, 20, 10],
    "activations": ['leaky_relu', 'leaky_relu', 'softmax'],
}

mnist_network = Network(architecture)

print(mnist_network.calculate_parameter_count())