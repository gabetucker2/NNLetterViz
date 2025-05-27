# import library
import numpy as np

# import scripts
import debug
import param_configs.params_test as params
import functions.fwd_prop_funcs as fwd_prop_funcs
import functions.axon_funcs as axon_funcs
import functions.activation_funcs as activation_funcs

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

    ΔW_XY = params.μ * np.outer(Y, X)

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

    ΔW_XY = params.μ * np.outer(Y_norm, X_norm)

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

    ΔW_XY = params.μ * np.outer(T, X)
    
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
    ΔW_XY = params.μ * np.outer(T, X_norm)
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
    X = np.array(X)
    Y = np.array(Y)
    T = np.array(T)
    E = T - Y
    ΔW_XY = params.μ * np.outer(E, X)
    return axon_funcs.clip_weights(W_XY + ΔW_XY)

def widrow_hoff_learning_deep(X, W_XY_matrix, Y=None, T=None):
    debug.log.indent_level += 1
    try:
        # Forward propagate if no activations are passed in
        if Y is None:
            Y = fwd_prop_funcs.fwd_prop_deep(X, W_XY_matrix, return_all_layers=True)

        # Format inputs
        T = np.array(T).reshape(-1)
        Activations = [np.array(y) for y in Y]
        Inputs = Activations[:-1]
        num_layers = len(W_XY_matrix)

        # Validate that each weight layer has a corresponding activation transition
        if len(Activations) != num_layers + 1:
            debug.log.error(f"[LAYER COUNT ERROR] Got {len(Activations)} activations but expected {num_layers + 1} for {num_layers} weight layers.")
            return W_XY_matrix

        # Initialize delta array
        Deltas = [None] * num_layers

        # Output layer delta
        A_out = Activations[-1]
        d_out = params.back_activation_function(A_out)
        error = T - A_out
        Deltas[-1] = error * d_out

        # Backpropagate deltas
        for l in reversed(range(num_layers - 1)):
            W_next = W_XY_matrix[l + 1]
            Δ_next = Deltas[l + 1]

            A_l = Activations[l + 1]
            d_l = np.array([params.back_activation_function(y) for y in A_l])

            backprop_signal = W_next.T @ Δ_next

            if backprop_signal.shape != d_l.shape:
                debug.log.error(f"[SHAPE ERROR] Layer {l}: backprop_signal {backprop_signal.shape} vs activation derivative {d_l.shape}")
                return W_XY_matrix

            Deltas[l] = backprop_signal * d_l

        # Update weights
        W_matrix_new = []
        for i in range(num_layers):
            a_in = Inputs[i].reshape(-1)
            delta = Deltas[i].reshape(-1)
            ΔW = params.μ * np.outer(delta, a_in)

            expected_shape = W_XY_matrix[i].shape
            if ΔW.shape != expected_shape:
                debug.log.error(f"Layer {i}: ΔW shape {ΔW.shape} does not match expected weight shape {expected_shape}")
                debug.log.error(f"Input shape: {a_in.shape}, Delta shape: {delta.shape}")
                return W_XY_matrix

            W_updated = W_XY_matrix[i] + ΔW
            W_clipped = axon_funcs.clip_weights(W_updated)
            W_matrix_new.append(W_clipped)

        return W_matrix_new

    finally:
        debug.log.indent_level -= 1
        