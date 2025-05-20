# import library
import numpy as np

# import scripts
import debug
import param_configs.params_test as params
import functions.fwd_prop_funcs as fwd_prop_funcs
import functions.axon_funcs as axon_funcs

# general functions
def learn(X, W, Y=None, T=None):
    """
    output: W_new
    """
    return params.learning_algorithm(X, W, Y=Y, T=T)

def learn_deep(X, W_matrix, Y=None, T=None):
    """
    output: W_matrix_new
    """
    return params.learning_algorithm_deep(X, W_matrix, Y=Y, T=T)

# specific learning functions
def unsup_hebbian_learning(X, W, Y=None, T=None):
    """
    output: W_new
    """
    X = np.array(X)
    Y = np.array(Y)

    ΔW = params.μ * np.outer(Y, X)

    W_new = W + ΔW

    return axon_funcs.clip_weights(W_new)

def unsup_hebbian_learning_deep(X, W_matrix, Y=None, T=None):
    """
    output: W_matrix_new
    """
    X_working = np.array(X.copy())
    W_matrix_new = []

    debug.log.indent_level += 1
    for i, W in enumerate(W_matrix):
        Y_layer = np.array([fwd_prop_funcs.fwd_prop_single(X_working, w_row) for w_row in W])
        W_new = unsup_hebbian_learning(X_working, W, Y=Y_layer)
        W_matrix_new.append(W_new)
        X_working = Y_layer
    debug.log.indent_level -= 1

    return W_matrix_new

def unsup_norm_hebbian_learning(X, W, Y=None, T=None):
    """
    output: W_new
    """
    X = np.array(X)
    Y = np.array(Y)

    X_norm = axon_funcs.euc_normalize_membrane_pots(X)
    Y_norm = axon_funcs.euc_normalize_membrane_pots(Y)

    ΔW = params.μ * np.outer(Y_norm, X_norm)

    W_new = W + ΔW

    return axon_funcs.clip_weights(W_new)

def unsup_norm_hebbian_learning_deep(X, W_matrix, Y=None, T=None):
    """
    output: W_matrix_new
    """
    X_working = np.array(X.copy())
    W_matrix_new = []

    debug.log.indent_level += 1
    for i, W in enumerate(W_matrix):
        Y_layer = np.array([fwd_prop_funcs.fwd_prop_single(X_working, w_row) for w_row in W])
        W_new = unsup_norm_hebbian_learning(X_working, W, Y=Y_layer)
        W_matrix_new.append(W_new)
        X_working = Y_layer
    debug.log.indent_level -= 1

    return W_matrix_new

def semisup_hebbian_learning(X, W, Y=None, T=None):
    """
    output: W_new
    """
    X = np.array(X)
    T = np.array(T)

    ΔW = params.μ * np.outer(T, X)
    
    W_new = W + ΔW

    return axon_funcs.clip_weights(W_new)

def semisup_hebbian_learning_deep(X, W_matrix, Y=None, T=None):
    """
    output: W_matrix_new
    """
    X_working = np.array(X.copy())
    W_matrix_new = []

    debug.log.indent_level += 1
    for i, W in enumerate(W_matrix):
        if i < len(W_matrix) - 1:
            Y_layer = np.array([fwd_prop_funcs.fwd_prop_single(X_working, w_row) for w_row in W])
            W_new = unsup_hebbian_learning(X_working, W, Y=Y_layer)
            X_working = Y_layer
        else:
            W_new = semisup_hebbian_learning(X_working, W, T=T)
        W_matrix_new.append(W_new)
    debug.log.indent_level -= 1

    return W_matrix_new

def semisup_norm_hebbian_learning(X, W, Y=None, T=None):
    """
    output: W_new
    """
    X = np.array(X)
    T = np.array(T)

    X_norm = X / (np.linalg.norm(X) + 1e-8)
    ΔW = params.μ * np.outer(T, X_norm)
    W_new = W + ΔW

    return axon_funcs.clip_weights(W_new)

def semisup_norm_hebbian_learning_deep(X, W_matrix, Y=None, T=None):
    """
    output: W_matrix_new
    """
    X_working = np.array(X.copy())
    W_matrix_new = []

    debug.log.indent_level += 1
    for i, W in enumerate(W_matrix):
        if i < len(W_matrix) - 1:
            Y_layer = np.array([fwd_prop_funcs.fwd_prop_single(X_working, w_row) for w_row in W])
            W_new = unsup_norm_hebbian_learning(X_working, W, Y=Y_layer)
            X_working = Y_layer
        else:
            W_new = semisup_norm_hebbian_learning(X_working, W, T=T)
        W_matrix_new.append(W_new)
    debug.log.indent_level -= 1

    return W_matrix_new

def widrow_hoff_learning(X, W, Y=None, T=None):
    """
    output: W_new
    """
    X = np.array(X)
    Y = np.array(Y)
    T = np.array(T)

    dY = params.activation_function(Y, derivative=True, from_output=True)
    Delta = (T - Y) * dY
    ΔW = params.μ * np.outer(Delta, X)

    W_new = np.array(W) + ΔW

    return axon_funcs.clip_weights(W_new)

def widrow_hoff_learning_deep(X, W_matrix, Y=None, T=None):
    """
    output: W_matrix_new
    """
    if Y is None:
        Y = fwd_prop_funcs.fwd_prop_deep(X, W_matrix, return_all_layers=True)

    Activations = [np.array(X)] + [np.array(Y_i) for Y_i in Y]
    num_layers = len(W_matrix)
    Deltas = [None] * num_layers

    Y_output = np.array(Y[-1])
    T = np.array(T)
    Act_deriv_output = np.array([params.activation_function(Y_i, derivative=True, from_output=True) for Y_i in Y_output])
    Deltas[-1] = (T - Y_output) * Act_deriv_output

    for l in reversed(range(num_layers - 1)):
        W_next = np.array(W_matrix[l + 1])
        Δnext = Deltas[l + 1]
        Y_l = np.array(Y[l])
        Act_deriv = np.array([params.activation_function(Y_i, derivative=True, from_output=True) for Y_i in Y_l])
        Deltas[l] = (W_next.T @ Δnext) * Act_deriv

    W_matrix_new = []
    for i in range(num_layers):
        Input_to_layer = Activations[i]
        W = np.array(W_matrix[i])
        Error_signal = Deltas[i]
        ΔW = params.μ * np.outer(Error_signal, Input_to_layer)
        W_new = W + ΔW
        W_matrix_new.append(W_new)

    return W_matrix_new
