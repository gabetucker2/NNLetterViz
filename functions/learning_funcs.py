# import library
import numpy as np

# import scripts
import debug
import param_configs.params_test as params
import functions.math_funcs as math_funcs
import functions.fwd_prop_funcs as fwd_prop_funcs

def learn(x, w, y=None, t=None):
    """
    output: w_new
    """
    params.learning_algorithm(x, w, y=y, t=t)

def learn_deep(x, w_matrix, y=None, t=None):
    """
    output: w_new
    """
    params.learning_algorithm_deep(x, w_matrix, y=y, t=t)

def unsup_hebbian_learning(x, w, y=None, t=None):
    """
    output: w_new
    """
    x = np.array(x)
    y = np.array(y)

    x_norm = x / (np.linalg.norm(x) + 1e-8)
    y_norm = y / (np.linalg.norm(y) + 1e-8)

    delta_w = params.μ * np.outer(y_norm, x_norm)
    w_new = w + delta_w

    return np.clip(w_new, -10, 10)

def unsup_hebbian_learning_deep(x, w_matrix, y=None, t=None):
    """
    output: w_matrix_new
    """
    x_working = np.array(x.copy())
    w_matrix_new = []

    debug.log.indent_level += 1
    for i, w in enumerate(w_matrix):
        y_layer = np.array([fwd_prop_funcs.fwd_prop_single(x_working, w_row) for w_row in w])
        w_new = unsup_hebbian_learning(x_working, w, y=y_layer)
        w_matrix_new.append(w_new)
        x_working = y_layer
    debug.log.indent_level -= 1

    return w_matrix_new

def sup_hebbian_learning(x, w, y=None, t=None):
    """
    output: w_new
    """
    x = np.array(x)
    t = np.array(t)

    x_norm = x / (np.linalg.norm(x) + 1e-8)

    w_new = w.copy()
    delta_w = params.μ * np.outer(t, x_norm)
    w_new += delta_w

    return np.clip(w_new, -5, 5)

def sup_hebbian_learning_deep(x, w_matrix, y=None, t=None):
    """
    output: w_matrix_new
    """
    x_working = np.array(x.copy())
    w_matrix_new = []

    debug.log.indent_level += 1
    for i, w in enumerate(w_matrix):
        if i < len(w_matrix) - 1:
            y_layer = np.array([fwd_prop_funcs.fwd_prop_single(x_working, w_row) for w_row in w])
            w_new = unsup_hebbian_learning(x_working, w, y=y_layer)
            x_working = y_layer
        else:
            w_new = sup_hebbian_learning(x_working, w, t=t)
        w_matrix_new.append(w_new)
    debug.log.indent_level -= 1

    return w_matrix_new

def widrow_hoff_learning(x, w, y=None, t=None):
    """
    output: w_new
    """
    x = np.array(x)
    y = np.array(y)
    t = np.array(t)

    dy = params.activation_function(y, derivative=True, from_output=True)
    delta = (t - y) * dy
    delta_w = params.μ * np.outer(delta, x)

    w_new = np.array(w) + delta_w
    return w_new

def widrow_hoff_learning_deep(x, w_matrix, y=None, t=None):
    """
    output: w_matrix_new
    """
    if y is None:
        y = fwd_prop_funcs.fwd_prop_deep(x, w_matrix, return_all_layers=True)

    activations = [np.array(x)] + [np.array(y_i) for y_i in y]
    num_layers = len(w_matrix)
    deltas = [None] * num_layers

    y_output = np.array(y[-1])
    t = np.array(t)
    act_deriv_output = np.array([params.activation_function(y_i, derivative=True, from_output=True) for y_i in y_output])
    deltas[-1] = (t - y_output) * act_deriv_output

    for l in reversed(range(num_layers - 1)):
        w_next = np.array(w_matrix[l + 1])
        delta_next = deltas[l + 1]
        y_l = np.array(y[l])
        act_deriv = np.array([params.activation_function(y_i, derivative=True, from_output=True) for y_i in y_l])
        deltas[l] = (w_next.T @ delta_next) * act_deriv

    w_matrix_new = []
    for i in range(num_layers):
        input_to_layer = activations[i]
        w = np.array(w_matrix[i])
        error_signal = deltas[i]
        delta_w = params.μ * np.outer(error_signal, input_to_layer)
        w_new = w + delta_w
        w_matrix_new.append(w_new)

    return w_matrix_new
