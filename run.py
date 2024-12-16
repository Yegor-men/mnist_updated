from real_data import X

import pickle

import numpy as np

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

def print_architecture():
    print(f"Parameters: {model.calculate_parameter_count()}\n")
    for i in range(len(model.layers)):
        print(f"Layer {i}: {model.layers[i]} nodes, {model.activations[i]} activation")

print_architecture()