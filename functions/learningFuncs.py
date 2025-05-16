# import library
import numpy as np

# import scripts
import debug
import paramConfigs.paramsTest as params
import functions.mathFuncs as mathFuncs
import functions.fwdPropFuncs as fwdPropFuncs

def learn(X, Y, W, T = []):
    """
    output: W_new
    """
    params.learningAlgorithm(X, Y, W, T)

def learnDeep(X, Y, W_matrix, T = []):
    """
    output: W_new
    """
    params.learningAlgorithmDeep(X, Y, W_matrix, T)

def hebbianLearning(X, Y, W, T = []):
    """
    output: W_new
    """
    debug.Log.backProp(f"Beginning Hebbian Learning...")

    ΔW = params.μ * np.array(X) * np.array(Y)

    W_new = np.array(W) + ΔW

    debug.Log.backProp(f"Returning final layer's updated weights: {W_new}")
    return W_new

def hebbianLearningDeep(X, Y, W_matrix, T = []):
    """
    output: W_matrix_new
    """
    debug.Log.backProp(f"Beginning deep Hebbian Learning...")

    X_working = np.array(X.copy())
    W_matrix_new = []

    debug.Log.indent_level += 1
    for i, W in enumerate(W_matrix):
        debug.Log.axons(f"Performing deep Hebbian learning on axon layer {i}")

        W_new = hebbianLearning(X, Y, W)

        W_matrix_new.append(W_new)

        X_working = fwdPropFuncs.fwdProp(X_working, W)

    debug.Log.indent_level -= 1

    debug.Log.backProp(f"Returning updated deep weight matrix...")
    return W_matrix_new

def widrowHoffLearning(X, Y, W, T):
    """
    output: W_new
    """
    debug.Log.backProp(f"Beginning Widrow-Hoff Learning...")

    ΔW = params.μ * (T - Y) * X

    W_new = W + ΔW

    debug.Log.backProp(f"Returning updated weight matrix...")
    return W_new

def widrowHoffLearningDeep(X, Y, W_matrix, T):
    """
    output: W_matrix_new
    """
    debug.Log.backProp("Beginning Widrow-Hoff Learning...")

    W_matrix_new = []

    X_working = X

    debug.Log.indent_level += 1
    for i in range(len(W_matrix)):
        debug.Log.axons(f"Performing deep Widrow Hoff on axon layer {i}")
        W = W_matrix[i]
        Y_layer = Y[i]

        target = T if i == len(W_matrix) - 1 else Y_layer

        ΔW = params.μ * np.outer((target - Y_layer), X_working)
        W_new = W + ΔW
        W_matrix_new.append(W_new)

        debug.Log.backProp(f"Layer {i}: ΔW = {ΔW}")
        debug.Log.backProp(f"Layer {i}: Updated weights = {W_new}")

        X_working = Y_layer

    debug.Log.indent_level += 1

    debug.Log.backProp("Returning updated weight matrix...")
    return W_matrix_new
    