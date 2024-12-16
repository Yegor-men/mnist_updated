from training_data import trainX, trainY, testX, testY
from activations import *

import pickle

import numpy as np
np.random.seed(1)
random.seed(1)


class Network:
    def __init__(self, architecture):
        self.architecture = architecture
        self.input_size = architecture["layer_neurons"][0]
        self.output_size = architecture["layer_neurons"][-1]
        
        self.layers = architecture["layer_neurons"]
        self.activations = architecture["layer_activations"]

    def calculate_parameter_count(self):
        params = 0
        for i in range(0, len(self.layers) - 1):
            params += self.layers[i] * self.layers[i+1]
        params += sum(self.layers[1:-1])
        return params

    def create(self):
        self.biases = []
        self.weights = []

        for i in range(len(self.layers)):
            biases.append (np.zeros((self.layers[i], 1)))
            weights.append (np.zeros((self.layers[i], self.layers[i+1])))
    
    def forward_pass(self):
        pass

    def backward_pass(self):
        pass

    def save_parameters(self):
        with open("model.pkl", "wb") as f:
            pickle.dump(self, f)












architecture = {
    "layer_neurons": [784, 20, 20, 10],
    "layer_activations": ['none', 'leaky_relu', 'leaky_relu', 'softmax'],
}

mnist_network = Network(architecture)

print(mnist_network.calculate_params())