# import library
import numpy as np

# import scripts
import debug
import paramConfigs.paramsTest as params
import functions.mathFuncs as mathFuncs
import functions.fwdPropFuncs as fwdPropFuncs

def learn(X, W, Y=None, T=None):
    """
    output: W_new
    """
    params.learningAlgorithm(X, W, Y=Y, T=T)

def learnDeep(X, W_matrix, Y=None, T=None):
    """
    output: W_new
    """
    params.learningAlgorithmDeep(X, W_matrix, Y=Y, T=T)

def unsupHebbianLearning(X, W, Y=None, T=None):
    """
    output: W_new
    """
    X = np.array(X)
    Y = np.array(Y)

    # Normalize to avoid runaway values
    X_norm = X / (np.linalg.norm(X) + 1e-8)
    Y_norm = Y / (np.linalg.norm(Y) + 1e-8)

    ΔW = params.μ * np.outer(Y_norm, X_norm)
    W_new = W + ΔW

    # Clip to prevent overflow
    return np.clip(W_new, -10, 10)

def unsupHebbianLearningDeep(X, W_matrix, Y=None, T=None):
    """
    output: W_matrix_new
    """
    # debug.log.backProp(f"Beginning deep Hebbian Learning...")

    X_working = np.array(X.copy())
    W_matrix_new = []

    debug.log.indent_level += 1
    for i, W in enumerate(W_matrix):
        Y_layer = np.array([fwdPropFuncs.fwdPropSingle(X_working, w) for w in W])
        W_new = unsupHebbianLearning(X_working, W, Y=Y_layer)
        W_matrix_new.append(W_new)
        X_working = Y_layer
    debug.log.indent_level -= 1

    return W_matrix_new

def supHebbianLearning(X, W, Y=None, T=None):
    """
    """
    X = np.array(X)
    T = np.array(T)

    # Normalize input to unit vector (important for stability)
    X_norm = X / (np.linalg.norm(X) + 1e-8)

    W_new = W.copy()

    # Update only the target class weights
    target_idx = np.argmax(T)
    ΔW = params.μ * np.outer(T, X_norm)  # T is one-hot, so this is a row update

    W_new += ΔW
    return np.clip(W_new, -5, 5)

def supHebbianLearningDeep(X, W_matrix, Y=None, T=None):
    """
    """
    X_working = np.array(X.copy())
    W_matrix_new = []

    debug.log.indent_level += 1
    for i, W in enumerate(W_matrix):
        if i < len(W_matrix) - 1:
            # Forward propagate normally through hidden layers
            Y_layer = np.array([fwdPropFuncs.fwdPropSingle(X_working, w) for w in W])
            W_new = unsupHebbianLearning(X_working, W, Y=Y_layer)
            X_working = Y_layer
        else:
            # Final layer: supervised Hebbian update using T
            W_new = supHebbianLearning(X_working, W, T=T)
        W_matrix_new.append(W_new)
    debug.log.indent_level -= 1

    return W_matrix_new

def widrowHoffLearning(X, W, Y=[], T=[]):
    """
    output: W_new
    """
    X = np.array(X)
    Y = np.array(Y)
    T = np.array(T)

    # Compute the derivative of the activation function at the output
    dY = params.activationFunction(Y, derivative=True, from_output=True)

    # Compute delta: element-wise (T - Y) * f'(Y)
    delta = (T - Y) * dY

    # Weight update using outer product
    ΔW = params.μ * np.outer(delta, X)

    W_new = np.array(W) + ΔW
    return W_new

def widrowHoffLearningDeep(X, W_matrix, Y=[], T=[]):
    """
    output: W_matrix_new
    """

    # Forward pass if needed
    if not Y:
        Y = fwdPropFuncs.fwdPropDeep(X, W_matrix, return_all_layers=True)

    A = [np.array(X)] + [np.array(y) for y in Y]  # Activations at each layer
    L = len(W_matrix)

    δ = [None] * L  # Deltas for each layer

    # Compute delta at output layer
    Y_output = np.array(Y[-1])
    T = np.array(T)
    act_deriv_output = np.array([params.activationFunction(y, derivative=True, from_output=True) for y in Y_output])
    δ[-1] = (T - Y_output) * act_deriv_output

    # Backpropagate to hidden layers
    for l in reversed(range(L - 1)):
        W_next = np.array(W_matrix[l + 1])
        δ_next = δ[l + 1]
        Y_l = np.array(Y[l])
        act_deriv = np.array([params.activationFunction(y, derivative=True, from_output=True) for y in Y_l])
        δ[l] = (W_next.T @ δ_next) * act_deriv

    # Update weights layer-by-layer
    W_matrix_new = []
    for i in range(L):
        input_to_layer = A[i]
        W = np.array(W_matrix[i])
        error_signal = δ[i]
        ΔW = params.μ * np.outer(error_signal, input_to_layer)
        W_new = W + ΔW
        W_matrix_new.append(W_new)

    return W_matrix_new
