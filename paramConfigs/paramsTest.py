# import scripts
import functions.learningFuncs as learningFuncs
import functions.activationFuncs as activationFuncs
import functions.noiseFuncs as noiseFuncs

# ---STATIC

# hiddenLayers
n_hiddenLayers = 5
n_hidden = 10

# outputs
n_outputs = 1 # do not change yet
axonPotInterference = 0

# functions
learningAlgorithm = learningFuncs.hebbianLearning
learningAlgorithmDeep = learningFuncs.hebbianLearningDeep

activationFunction = activationFuncs.activationFunction_sigmoid
noiseFunction = noiseFuncs.norm_noise

# learning parameters
n_epochs = 5
μ = 0.01
