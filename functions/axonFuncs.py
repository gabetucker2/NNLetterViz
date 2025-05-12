# script imports
import paramConfigs.paramsTest as params

# library imports
import random

# axonPotentialFuncs
def c_axonPotential(c_pre, a_preToPost):
    return c_pre * a_preToPost

def c_postActivation(C_axonPotential):
    return params.activationFunction(sum(C_axonPotential))

def a_initAxonWeight():
    return random.random()
