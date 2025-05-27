# library imports
import numpy as np

# functions
def activation_linear(x):
    x = np.array(x)
    return x

def activation_linear_derivative(x):
    x = np.array(x)
    return np.ones_like(x)

def activation_threshold(x):
    x = np.array(x)
    return (x >= 0.5).astype(float)

def activation_threshold_derivative(x):
    x = np.array(x)
    return np.zeros_like(x)

def activation_sigmoid(x):
    x = np.array(x)
    return 1 / (1 + np.exp(-x))

def activation_sigmoid_derivative(x, from_output=False):
    x = np.array(x)
    if from_output:
        return x * (1 - x)  # x is already sigmoid output
    s = 1 / (1 + np.exp(-x))
    return s * (1 - s)
