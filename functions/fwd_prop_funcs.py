# import library
import numpy as np

# import scripts
import param_configs.params_test as params
import debug

# functions
def fwd_prop(X, W_XY):
    """
    X: vector of presynaptic membrane potentials
    W_XY: matrix of axon conductances between the two layers (output_neurons x input_neurons)
    output: vector of postsynaptic membrane potentials
    """
    assert W_XY.shape[0] == X.shape[0], (
        f"[FWDPROP ERROR] Shape mismatch: X shape {X.shape} not compatible with W_XY shape {W_XY.shape}"
    )
    b = params.noise_function(params.axon_pot_interference)
    S = np.matmul(X, W_XY) + b
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
    for i, W_XY in enumerate(W_layers):
        assert W_XY.shape[0] == X_working.shape[0], (
            f"[FWDPROP DEEP ERROR] At layer {i}: X shape {X_working.shape} not compatible with W_XY shape {W_XY.shape}"
        )
        X_working = fwd_prop(X_working, W_XY)
        outputs.append(X_working)
    
    return outputs if return_all_layers else outputs[-1]
