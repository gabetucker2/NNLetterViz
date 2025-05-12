# import scripts
import functions.mathFuncs as mathFuncs
import functions.fwdPropFuncs as fwdPropFuncs

# functions
def hebbianLearning(C_trainVec, ):
    C_hiddenLayer1 = fwdPropFuncs.C_fwdProp(C_trainVec, ninput_nhidden_a)

    C_nhiddenLayers_hiddenLayerX = [C_hiddenLayer1]
    for i_hiddenLayer in range(1, params.n_hiddenLayers):
        C_hiddenLayerPre = C_nhiddenLayers_hiddenLayerX[i_hiddenLayer - 1]
        C_hiddenLayerPost = fwdPropFuncs.C_fwdProp(
            C_hiddenLayerPre, nhiddenLayers_nhidden_nhidden_a[i_hiddenLayer]
        )
        C_nhiddenLayers_hiddenLayerX.append(C_hiddenLayerPost)

    C_output = fwdPropFuncs.C_fwdProp(C_nhiddenLayers_hiddenLayerX[-1], nhidden_noutput_a)
    c_output = C_output[0]
