# script imports
import debug
import data.letterData as letterData
import paramConfigs.paramsTest as params
import functions.activationFuncs as activationFuncs
import functions.mathFuncs as mathFuncs
import functions.fwdPropFuncs as fwdPropFuncs

# library imports
import math
import random

debug.Log.indent_level = 0
debug.Log.procedure(f"Setting up variables...")

numTestingEpochs = 10

I = len(letterData.letterVariants[0][0]) # fetch bits per letter
HN = 6 # neurons per hidden layer
HM = 3 # num hidden layers
O = len(letterData.letterVariants) # fetch number of letters

trainTestSplitRatio = 0.5

debug.Log.training(f"Training model...")



debug.Log.testing(f"Testing model...")

for epoch in range(numTestingEpochs):
    debug.Log.epoch(f"Epoch {epoch} / {numTestingEpochs}")

    debug.Log.epoch("Splitting and randomizing order of data for training and testing...")
    trainingData, testingData = mathFuncs.shuffle_split(letterData.letterVariants, trainTestSplitRatio)

    debug.Log.epoch("Initializing axon conductances...")
    axons_NI = [[1]*I for _ in range(HN)] # 2D
    axons_HMHN = [[[1]*HN for _ in range(HN)] for _ in range(HM)] # 3D
    axons_NO = [[1]*O for _ in range(HN)] # 2D
    axons_allLayers = [axons_NI,
                        *axons_HMHN, # unpack into HM axons_HN rows
                        axons_NO]

    debug.Log.testing(f"Beginning tests...")
    numTests = 0
    numAccuratePredictions = 0

    debug.Log.indent_level += 1
    for actualLetter, matrices in letterData.letterVariants.items():
        debug.Log.testing(f"Testing instances of letter type \'{actualLetter}\'")
        numTests += 1
        OMemPots = [] # output membrane potentials for this letter

        debug.Log.indent_level += 1
        for i, matrix in enumerate(matrices):
            debug.Log.testing(f"Iterating over letter \'{actualLetter}\' instance {i}")
            OMemPot = fwdPropFuncs.fwdProp(mathFuncs.matrix_to_vector(matrix), axons_allLayers)
            OMemPots.append(OMemPot)

        debug.Log.indent_level -= 1

        predictedLetter = math.max(OMemPots)
        if predictedLetter == actualLetter:
            numAccuratePredictions += 1

        debug.Log.testing(f"Letter \'{actualLetter}\' outputted membrane potentials: {OMemPots}")
        debug.Log.testing(f"Model prediction: \'{predictedLetter}\'; Actual: \'{actualLetter}\'")

    debug.Log.indent_level -= 1

    debug.Log.testing(f"Model accuracy: {numTests / numAccuratePredictions:.2%}")
