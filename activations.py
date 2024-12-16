class Activation:
    def __init__(self, function):
        self.activation_function = function

    def apply(self, data):
        if self.activation_function == "none":
            return data
        elif self.activation_function == 'relu':
            return np.maximum(data, 0)
        elif self.activation_function == 'leaky_relu':
            return np.maximum(0.01 * data, data)
        elif self.activation_function == 'sigmoid':
            return 1 / (1 + np.exp(-data))
        elif self.activation_function == 'tanh':
            return np.tanh(data)
        elif self.activation_function == 'softmax':
            exp_data = np.exp(data)
            return exp_data / np.sum(exp_data)
    
    def unapply(self, data):
        if self.activation_function == "none":
            return data
        elif self.activation_function == 'relu':
            return np.where(data > 0, data, 0)
        elif self.activation_function == 'leaky_relu':
            return np.where(data > 0, data, 0.01 * data)
        elif self.activation_function == 'sigmoid':
            return np.log(data / (1 - data))
        elif self.activation_function == 'tanh':
            return np.arctanh(data)
        elif self.activation_function == 'softmax':
            # Note: unapplying softmax is not well-defined, as it's not invertible
            # This implementation returns the input data unchanged
            return data


none = Activation('none')
relu = Activation('relu')
leaky_relu = Activation('leaky_relu')
sigmoid = Activation('sigmoid')
tanh = Activation('tanh')
softmax = Activation('softmax')