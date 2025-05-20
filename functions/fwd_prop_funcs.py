# import library
import numpy as np

# import scripts
import param_configs.params_test as params

# functions
def fwd_prop(X, W_matrix):
    """
    X: vector of presynaptic membrane potentials
    W_matrix: matrix of axon conductances between the two layers (output_neurons x input_neurons)
    output: vector of postsynaptic membrane potentials
    """
    b = params.noise_function(params.axon_pot_interference)
    S = np.dot(X, W_matrix) + b
    Y = params.activation_function(S)
    
    return Y

def fwd_prop_deep(X, W_layers, return_all_layers=False):
    """
    X: vector of input membrane potentials
    W_layers: list of matrices of axon conductances between layers (layers × output_neurons × input_neurons)
    return_all_layers: whether to return the full propagation trace or just the final output
    output: matrix of postsynaptic membrane potentials for all layers if return_all_layers=True
    output: vector of the final postsynaptic membrane potentials in the matrix if return_all_layers=False
    """
    X_working = X
    outputs = [X_working]
    for W_matrix in W_layers:
        Y = fwd_prop(X, W_matrix)
        X_working = Y
        outputs.append(Y)

    return outputs if return_all_layers else outputs[-1]
