# import library
import numpy as np

# import scripts
import param_configs.params_test as params
import debug

# functions
def fwd_prop_vector(X, W_matrix):
    return np.array([params.activation_function(np.dot(w, X) + params.noise_function(params.axon_pot_interference)) for w in W_matrix])

def fwd_prop_deep(X, W_layers, return_all_layers=False):
    outputs = [np.array(X)]
    for W in W_layers:
        X = fwd_prop_vector(X, W)
        outputs.append(X)
    return outputs if return_all_layers else outputs[-1]
