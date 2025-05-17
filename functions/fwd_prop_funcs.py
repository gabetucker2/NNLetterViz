# import library
import numpy as np

# import scripts
import debug
import functions.math_funcs as math_funcs
import param_configs.params_test as params

def fwd_prop_single(x, w):
    """
    Single output neuron: dot product + noise + activation
    x: (n_inputs,)
    w: (n_inputs,)
    Returns: scalar output
    """
    s = np.dot(x, w) + params.noise_function(params.axon_pot_interference)
    y = params.activation_function(s)
    
    return y

def fwd_prop_vector(x, w_3d_matrix):
    """
    Layer-wise forward propagation.
    x: (n_inputs,)
    w_3d_matrix: (n_outputs, n_inputs) – one weight vector per output neuron
    Returns: (n_outputs,) – vector of activations
    """
    outputs = []
    debug.log.indent_level += 1
    for j, w_row in enumerate(w_3d_matrix):
        # debug.log.axons(f"Output neuron {j}")
        y_j = fwd_prop_single(x, w_row)
        outputs.append(y_j)
    debug.log.indent_level -= 1
    return np.array(outputs)

def fwd_prop_deep(x, w_4d_matrix, return_all_layers=False):
    x_working = x
    outputs = [x_working]  # Include input layer

    for w in w_4d_matrix:
        y = [params.activation_funcs.activation_function_sigmoid(np.dot(w_row, x_working)) for w_row in w]
        x_working = y
        outputs.append(y)

    return outputs if return_all_layers else outputs[-1]
