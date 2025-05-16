# import library
import numpy as np

# import scripts
import debug
import functions.mathFuncs as mathFuncs
import paramConfigs.paramsTest as params

def fwdProp(X, W):
    """
    output: Y
    """
    debug.Log.fwdProp(f"Beginning single-layer forward propagation...")

    S = np.dot(X, W) + params.noiseFunction(params.axonPotInterference)

    Y = params.activationFunction(S)

    debug.Log.fwdProp(f"Returning membrane potentials: {Y}")
    return Y


def fwdPropDeep(X, W_matrix):
    """
    output: Y (layer after the final W vector)
    """
    debug.Log.fwdProp(f"Beginning forward propagation on {len(W_matrix)} layers...")

    X_working = np.array([])
    Y_working = np.array(X.copy())
    debug.Log.indent_level += 1
    for i, W in enumerate(W_matrix):
        debug.Log.axons(f"Iterating through 3D axon layer {i}")

        X_working = Y_working
        Y_working = fwdProp(X_working, W)
            
    debug.Log.indent_level -= 1

    debug.Log.fwdProp(f"Returning final layer's membrane potentials: {Y_working}")
    return Y_working
