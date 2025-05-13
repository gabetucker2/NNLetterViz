# import scripts
import debug
import functions.mathFuncs as mathFuncs
import paramConfigs.paramsTest as params

def fwdProp(inputs, axons_allLayers):
    debug.Log.fwdProp(f"Beginning forward propagation...")

    numLayersToCompute = len(axons_allLayers)
    debug.Log.fwdProp(f"FwdProp to be performed on {numLayersToCompute} layers...")

    # post contains the input layer, which will immediately transfer to presynaptic layer upon booting up the loop
    preSynMemPots = [] # presynaptic membrane potentials
    postSynMemPots = inputs.copy() # postsynaptic membrane potentials
    debug.Log.indent_level += 1
    for i, axonLayer in enumerate(axons_allLayers):
        debug.Log.axons(f"Iterating through 3D axon layer {i}")
        debug.Log.axons(f"Transferring postSynMemPots data to preSynMemPost")
        preSynMemPots = postSynMemPots
        postSynMemPots = []

        debug.Log.indent_level += 1
        for j, axonConMatrix in enumerate(axonLayer): # axon conductance factor matrix for each presynaptic neuron
            debug.Log.axons(f"Iterating through 2D presynaptic axon conductance Matrix {j}")

            # aggregate axon potentials = a1w1 + a2w2 + ... + anwn + b
            x = mathFuncs.dot_product(preSynMemPots, axonConMatrix) + params.noiseFunction(params.axonPotInterference)
            # pass it through the activation function to get a postsynaptic membrane potential
            postSynMemPot = params.activationFunction(x)

            postSynMemPots.append(postSynMemPot)
            
        debug.Log.indent_level -= 1
    debug.Log.indent_level -= 1

    debug.Log.fwdProp(f"Returning final layer's membrane potentials: {postSynMemPots}")
    return postSynMemPots
