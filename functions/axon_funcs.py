# import scripts
import param_configs.params_test as params

# import libraries
import numpy as np

# import libraries
import random

def init_axon_val():
    return random.normalvariate(mu=params.init_axon_mean, sigma=params.init_axon_sd)

def get_axons(I, HN, HM, O):
    def new_layer(n_in, n_out):
        return np.array([[init_axon_val() for _ in range(n_out)] for _ in range(n_in)])
    
    # perceptron case
    if HM == 0:
        return [new_layer(I, O)]  # input × output

    # input → first hidden
    axons_input_hidden = new_layer(I, HN)  # I × HN

    # hidden → hidden
    axons_hidden_hidden = [new_layer(HN, HN) for _ in range(HM - 1)]

    # last hidden → output
    axons_hidden_output = new_layer(HN, O)  # HN × O

    return [axons_input_hidden] + axons_hidden_hidden + [axons_hidden_output]

def clip_weights(W_new):
    return np.clip(W_new, -params.axon_weight_max_dev, params.axon_weight_max_dev)

def euc_normalize_membrane_pots(X):
    return X / (np.sqrt(np.sum(np.abs(X) ** 2)) + 1e-8)
