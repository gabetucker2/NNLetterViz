# import library
import numpy as np

# import scripts
import param_configs.params_test as params
import functions.activation_funcs as activation_funcs
import functions.axon_funcs as axon_funcs

# functions
def fwd_prop_deep(X, W_4DMatrix, return_all_layers=False):
    X_working = X
    outputs = [X_working]  # Include input layer

    for W in W_4DMatrix:
        Y = [params.fwd_activation_function(np.dot(w_row, X_working)) for w_row in W]
        X_working = Y
        outputs.append(axon_funcs.clip_weights(Y))

    return outputs if return_all_layers else outputs[-1]

def fwd_prop(X, W_matrix):
    b = params.noise_function(params.axon_pot_interference)
    S = np.dot(np.transpose(X), np.transpose(W_matrix)) + b
    Y = params.fwd_activation_function(S)
    
    return Y
