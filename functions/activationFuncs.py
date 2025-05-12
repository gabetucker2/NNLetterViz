# library imports
import math

# activation functions
def activationFunction_linear(x):
    return x

def activationFunction_threshold(x):
    return x < 0.5

def activationFunction_sigmoid(x):
    return 1 / (1 + math.exp(-x))
