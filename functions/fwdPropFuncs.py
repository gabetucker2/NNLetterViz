# import library
import numpy as np

# import scripts
import debug
import functions.mathFuncs as mathFuncs
import paramConfigs.paramsTest as params

def fwdPropSingle(X, W):
    """
    Single output neuron: dot product + noise + activation
    X: (n_inputs,)
    W: (n_inputs,)
    Returns: scalar output
    """
    # debug.log.fwdProp(f"Computing single neuron potential...")
    S = np.dot(X, W) + params.noiseFunction(params.axonPotInterference)
    Y = params.activationFunction(S)
    # debug.log.fwdProp(f"Fwd prop output: {Y}")
    return Y

def fwdPropVector(X, W_3DMatrix):
    """
    Layer-wise forward propagation.
    X: (n_inputs,)
    W_3DMatrix: (n_outputs, n_inputs) – one weight vector per output neuron
    Returns: (n_outputs,) – vector of activations
    """
    # debug.log.fwdProp(f"Computing layer with {len(W_3DMatrix)} output neurons...")
    outputs = []
    debug.log.indent_level += 1
    for j, W in enumerate(W_3DMatrix):
        # debug.log.axons(f"Output neuron {j}")
        Yj = fwdPropSingle(X, W)
        outputs.append(Yj)
    debug.log.indent_level -= 1
    return np.array(outputs)

def fwdPropDeep(X, W_4DMatrix, return_all_layers=False):
    X_working = X
    outputs = [X_working]  # Include input layer

    for W in W_4DMatrix:
        Y = [params.activationFuncs.activationFunction_sigmoid(np.dot(w_row, X_working)) for w_row in W]
        X_working = Y
        outputs.append(Y)

    return outputs if return_all_layers else outputs[-1]
