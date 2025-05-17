# import scripts
import functions.learningFuncs as learningFuncs
import functions.activationFuncs as activationFuncs
import functions.noiseFuncs as noiseFuncs

# meta
numEpochs = 1

# NN learning params
μ = 0.1
numHiddenLayers = 1
numNeuronsPerHiddenLayer = 16

# NN algorithms
learningAlgorithm = learningFuncs.supHebbianLearning
learningAlgorithmDeep = learningFuncs.supHebbianLearningDeep
activationFunction = activationFuncs.activationFunction_linear

# NN noise
axonPotInterference = 0
noiseFunction = noiseFuncs.normalNoise

# NN axon generation
initAxonMean = 0
initAxonSD = 0.3

# misc
maxPrintRepeats = 99999
trainTestSplitRatio = 0.8
