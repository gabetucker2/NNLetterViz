# import scripts
import functions.learningFuncs as learningFuncs
import functions.activationFuncs as activationFuncs
import functions.noiseFuncs as noiseFuncs

# meta
numEpochs = 1
trainTestSplitRatio = 0.8

# NN learning params
μ = 0.01
numHiddenLayers = 2
numNeuronsPerHiddenLayer = 64

# NN algorithms
learningAlgorithm = learningFuncs.widrowHoffLearning
learningAlgorithmDeep = learningFuncs.widrowHoffLearningDeep
activationFunction = activationFuncs.activationFunction_linear

# NN noise
axonPotInterference = 0
noiseFunction = noiseFuncs.normalNoise

# NN axon generation
initAxonMean = 0
initAxonSD = 0.6

# misc
maxPrintRepeats = 99999
enableVisuals = True
