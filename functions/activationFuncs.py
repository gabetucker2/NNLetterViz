# library imports
import numpy as np

# functions
def activationFunction_linear(x, derivative=False, from_output=False):
    x = np.array(x)
    if derivative:
        return np.ones_like(x)
    return x

def activationFunction_threshold(x, derivative=False, from_output=False):
    x = np.array(x)
    if derivative:
        return np.zeros_like(x)  # undefined almost everywhere
    return (x >= 0.5).astype(float)

def activationFunction_sigmoid(x, derivative=False, from_output=False):
    x = np.array(x)
    x_clipped = np.clip(x, -500, 500)
    s = 1 / (1 + np.exp(-x_clipped))
    if derivative:
        return s * (1 - s) if not from_output else x * (1 - x)
    return s
