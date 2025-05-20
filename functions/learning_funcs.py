# import library
import numpy as np

# import scripts
import debug
import param_configs.params_test as params
import functions.fwd_prop_funcs as fwd_prop_funcs
import functions.axon_funcs as axon_funcs

# general functions
def learn(X, W_XY, Y=None, T=None):
    """
    output: W_new
    """
    return params.learning_algorithm(X, W_XY, Y=Y, T=T)

def learn_deep(X, W_XY_matrix, Y=None, T=None):
    """
    output: W_matrix_new
    """
    return params.learning_algorithm_deep(X, W_XY_matrix, Y=Y, T=T)

# specific learning functions
def unsup_hebbian_learning(X, W_XY, Y=None, T=None):
    """
    output: W_new
    """
    X = np.array(X)
    Y = np.array(Y)

    ΔW_XY = params.μ * np.outer(X, Y)

    W_new = W_XY + ΔW_XY

    return axon_funcs.clip_weights(W_new)

def unsup_hebbian_learning_deep(X, W_XY_matrix, Y=None, T=None):
    """
    output: W_matrix_new
    """
    X_working = np.array(X.copy())
    W_matrix_new = []

    debug.log.indent_level += 1
    for i, W_XY in enumerate(W_XY_matrix):
        Y_layer = fwd_prop_funcs.fwd_prop(X_working, W_XY)
        W_new = unsup_hebbian_learning(X_working, W_XY, Y=Y_layer)
        W_matrix_new.append(W_new)
        X_working = Y_layer
    debug.log.indent_level -= 1

    return W_matrix_new

def unsup_norm_hebbian_learning(X, W_XY, Y=None, T=None):
    """
    output: W_new
    """
    X = np.array(X)
    Y = np.array(Y)

    X_norm = axon_funcs.euc_normalize_membrane_pots(X)
    Y_norm = axon_funcs.euc_normalize_membrane_pots(Y)

    ΔW_XY = params.μ * np.outer(X_norm, Y_norm)

    W_new = W_XY + ΔW_XY

    return axon_funcs.clip_weights(W_new)

def unsup_norm_hebbian_learning_deep(X, W_XY_matrix, Y=None, T=None):
    """
    output: W_matrix_new
    """
    X_working = np.array(X.copy())
    W_matrix_new = []

    debug.log.indent_level += 1
    for i, W_XY in enumerate(W_XY_matrix):
        Y_layer = fwd_prop_funcs.fwd_prop(X_working, W_XY)
        W_new = unsup_norm_hebbian_learning(X_working, W_XY, Y=Y_layer)
        W_matrix_new.append(W_new)
        X_working = Y_layer
    debug.log.indent_level -= 1

    return W_matrix_new

def semisup_hebbian_learning(X, W_XY, Y=None, T=None):
    """
    output: W_new
    """
    X = np.array(X)
    T = np.array(T)

    ΔW_XY = params.μ * np.outer(X, T)
    
    W_new = W_XY + ΔW_XY

    return axon_funcs.clip_weights(W_new)

def semisup_hebbian_learning_deep(X, W_XY_matrix, Y=None, T=None):
    """
    output: W_matrix_new
    """
    X_working = np.array(X.copy())
    W_matrix_new = []

    debug.log.indent_level += 1
    for i, W_XY in enumerate(W_XY_matrix):
        if i < len(W_XY_matrix) - 1:
            Y_layer = fwd_prop_funcs.fwd_prop(X_working, W_XY)
            W_new = unsup_hebbian_learning(X_working, W_XY, Y=Y_layer)
            X_working = Y_layer
        else:
            W_new = semisup_hebbian_learning(X_working, W_XY, T=T)
        W_matrix_new.append(W_new)
    debug.log.indent_level -= 1

    return W_matrix_new

def semisup_norm_hebbian_learning(X, W_XY, Y=None, T=None):
    """
    output: W_new
    """
    X = np.array(X)
    T = np.array(T)

    X_norm = X / (np.linalg.norm(X) + 1e-8)
    ΔW_XY = params.μ * np.outer(X_norm, T)
    W_new = W_XY + ΔW_XY

    return axon_funcs.clip_weights(W_new)

def semisup_norm_hebbian_learning_deep(X, W_XY_matrix, Y=None, T=None):
    """
    output: W_matrix_new
    """
    X_working = np.array(X.copy())
    W_matrix_new = []

    debug.log.indent_level += 1
    for i, W_XY in enumerate(W_XY_matrix):
        if i < len(W_XY_matrix) - 1:
            Y_layer = fwd_prop_funcs.fwd_prop(X_working, W_XY)
            W_new = unsup_norm_hebbian_learning(X_working, W_XY, Y=Y_layer)
            X_working = Y_layer
        else:
            W_new = semisup_norm_hebbian_learning(X_working, W_XY, T=T)
        W_matrix_new.append(W_new)
    debug.log.indent_level -= 1

    return W_matrix_new

def widrow_hoff_learning(X, W_XY, Y=None, T=None):
    """
    output: W_new
    """
    X = np.array(X)
    Y = np.array(Y)
    T = np.array(T)

    E = T - Y

    ΔW_XY = params.μ * np.outer(X, E)

    W_new = W_XY + ΔW_XY

    return axon_funcs.clip_weights(W_new)

def widrow_hoff_learning_deep(X, W_XY_matrix, Y=None, T=None):
    if Y is None:
        Y = fwd_prop_funcs.fwd_prop_deep(X, W_XY_matrix, return_all_layers=True)

    Activations = [np.array(X)] + [np.array(y) for y in Y]
    num_layers = len(W_XY_matrix)
    Deltas = [None] * num_layers

    # Handle output layer delta explicitly
    A_last = Activations[-1]
    T = np.array(T)
    Act_deriv_last = params.activation_function(A_last, derivative=True, from_output=True)
    Deltas[-1] = (T - A_last) * Act_deriv_last

    # Recursive backpropagation for hidden layers
    def backprop_recursive(l):
        if l < 0:
            return

        A_l = Activations[l + 1]
        W_next = W_XY_matrix[l + 1]
        Δ_next = Deltas[l + 1]

        Act_deriv = params.activation_function(A_l, derivative=True, from_output=False)

        if W_next.shape[1] != Δ_next.shape[0]:
            debug.log.error(f"[LAYER {l}] Δ_next shape {Δ_next.shape} does not match W_next output dim {W_next.shape[1]}")
            return

        if W_next.shape[0] != A_l.shape[0]:
            debug.log.error(f"[LAYER {l}] activation shape {A_l.shape} does not match W_next input dim {W_next.shape[0]}")
            return

        Deltas[l] = (W_next @ Δ_next) * Act_deriv
        backprop_recursive(l - 1)

    backprop_recursive(num_layers - 2)  # Recurse from second-to-last layer

    # Weight updates
    W_matrix_new = []
    for i in range(num_layers):
        Input_to_layer = Activations[i]
        Error_signal = Deltas[i]
        W_XY = W_XY_matrix[i]

        if Input_to_layer.ndim != 1 or Error_signal.ndim != 1:
            debug.log.error(f"Inputs to outer product must be vectors. Got {Input_to_layer.shape}, {Error_signal.shape}")
            return W_XY_matrix

        ΔW_XY = params.μ * np.outer(Input_to_layer, Error_signal)
        W_new = W_XY + ΔW_XY
        W_matrix_new.append(axon_funcs.clip_weights(W_new))

    return W_matrix_new
