# library imports
import numpy as np

# functions
def activation_function_linear(x, derivative=False, from_output=False):
    x = np.array(x)
    if derivative:
        return np.ones_like(x)
    return x

def activation_function_threshold(x, derivative=False, from_output=False):
    x = np.array(x)
    if derivative:
        return np.zeros_like(x)
    return (x >= 0.5).astype(float)

def activation_function_sigmoid(x, derivative=False, from_output=False):
    x = np.array(x)
    s = 1 / (1 + np.exp(-x))
    if derivative:
        return s * (1 - s) if not from_output else x * (1 - x)
    return s
