# import library
import numpy as np

# import scripts
import param_configs.params_test as params
import functions.activation_funcs as activation_funcs
import debug

# functions
def fwd_prop_deep(X, W_4DMatrix, return_all_layers=False):
    X_working = X
    outputs = [X_working]  # Include input layer

    for W in W_4DMatrix:
        Y = [activation_funcs.activation_function_sigmoid(np.dot(w_row, X_working)) for w_row in W]
        X_working = Y
        outputs.append(Y)

    return outputs if return_all_layers else outputs[-1]