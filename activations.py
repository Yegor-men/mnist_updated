import numpy as np

class Activation:
    def __init__(self):
        pass

    def apply(self, data, function):
        if function == "none":
            return data
        elif function == 'relu':
            return np.maximum(data, 0)
        elif function == 'leaky_relu':
            return np.maximum(0.01 * data, data)
        elif function == 'sigmoid':
            return 1 / (1 + np.exp(-data))
        elif function == 'tanh':
            return np.tanh(data)
        elif function == 'softmax':
            exp_data = np.exp(data)
            return exp_data / np.sum(exp_data)
    
    def unapply(self, data, function):
        if function == "none":
            return data
        elif function == 'relu':
            return np.where(data > 0, 1, 0)
        elif function == 'leaky_relu':
            return np.where(data > 0, 1, 0.01)
        elif function == 'sigmoid':
            return data * (1 - data)
        elif function == 'tanh':
            return 1 - data ** 2
        elif function == 'softmax':
            return data

activation_function = Activation()