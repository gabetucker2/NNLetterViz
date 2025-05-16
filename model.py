# script imports
import debug
import data.letterData as letterData
import paramConfigs.paramsTest as params
import functions.mathFuncs as mathFuncs
import functions.fwdPropFuncs as fwdPropFuncs
import functions.axonFuncs as axonFuncs
import functions.learningFuncs as learningFuncs

# library imports
import math
from collections import defaultdict

debug.log.indent_level = 0
debug.log.procedure(f"Setting up variables...")

numEpochs = 10
I = len(next(iter(letterData.letterVariants.values()))[0])
HN = 6 # neurons per hidden layer
HM = 3 # num hidden layers
O = len(letterData.letterVariants) # fetch number of letters

trainTestSplitRatio = 0.5

epochLetterAccuracies = []

debug.log.procedure("Beginning simulation...")
for epoch in range(numEpochs):

    ##################################################################

    debug.log.epoch(f"Epoch {epoch} / {numEpochs}")

    debug.log.epoch("Splitting and randomizing order of data for training and testing...")
    trainingData, testingData = mathFuncs.shuffle_split(letterData.letterVariants, trainTestSplitRatio)

    debug.log.epoch("Initializing axon conductances...")
    W_matrix = axonFuncs.getAxons()

    ##################################################################

    debug.log.training(f"Beginning training...")
    debug.log.indent_level += 1
    for actualLetter, trainingMatrices in trainingData.items():
        debug.log.training(f"Training on matrix of letter type '{actualLetter}'")
        
        debug.log.indent_level += 1
        for i, trainingMatrix in enumerate(trainingMatrices):
            debug.log.training(f"Iterating over letter '{actualLetter}' instance {i}")
            X = mathFuncs.matrix_to_vector(trainingMatrix)
            Y = fwdPropFuncs.fwdPropDeep(X, W_matrix)
            W_matrix = learningFuncs.hebbianLearningDeep(X, Y, W_matrix)

        debug.log.indent_level -= 1
    debug.log.indent_level -= 1

    ##################################################################
    
    debug.log.testing(f"Beginning testing...")
    
    letterAccuracies = {}
    
    debug.log.indent_level += 1
    for actualLetter, testingMatrices in testingData.items():
        debug.log.testing(f"Testing matrix of letter type '{actualLetter}'")
        Y = [] # output membrane potentials for this letter

        debug.log.indent_level += 1
        for i, matrix in enumerate(testingMatrices):
            debug.log.testing(f"Iterating over letter '{actualLetter}' instance {i}")
            y = fwdPropFuncs.fwdPropDeep(mathFuncs.matrix_to_vector(matrix), W_matrix)
            Y.append(y)

        debug.log.indent_level -= 1

        predictedLetter = math.max(Y)
        if predictedLetter == actualLetter:
            accPredictionCounter += 1

        debug.log.testing(f"Letter '{actualLetter}' outputted membrane potentials: {Y}")
        debug.log.testing(f"Model prediction: '{predictedLetter}'; Actual: '{actualLetter}'")

        if actualLetter not in letterAccuracies:
            letterAccuracies[actualLetter] = []
        letterAccuracies[actualLetter].append(predictedLetter == actualLetter)

    debug.log.indent_level -= 1

    epochLetterAccuracies.append(letterAccuracies)

    ##################################################################

    newAccuracy = mathFuncs.compute_letter_accuracy(actualLetter, letterAccuracies)
    debug.log.analysis(f"Model accuracy in this epoch: {newAccuracy:.2%}")

    ##################################################################

debug.log.procedure("Beginning analytics...")

debug.log.procedure("Computing average accuracies across all epochs and letters...")
total_correct = 0
total_instances = 0
per_letter_results = defaultdict(list)

for epoch_data in epochLetterAccuracies:
    for letter, results in epoch_data.items():
        per_letter_results[letter].extend(results)
        total_correct += sum(results)
        total_instances += len(results)

average_accuracy = total_correct / total_instances if total_instances else 0.0

debug.log.procedure("Computing baseline random guess accuracy...")
num_letters = len(per_letter_results)
baseline_accuracy = 1 / num_letters if num_letters else 0.0

debug.log.procedure("Computing accuracy delta...")
accuracy_delta = average_accuracy - baseline_accuracy

##################################################################

debug.log.analysis("Final simulation analytics...")

debug.log.analysis(f"Number of epochs: {numEpochs}")

debug.log.analysis(f"Model random guess accuracy: {baseline_accuracy:.2%}")
debug.log.analysis(f"Average model accuracy across {numEpochs} epochs: {average_accuracy:.2%}")
debug.log.analysis(f"Model accuracy delta: {accuracy_delta:.2%}")

debug.log.analysis("Average accuracy per letter:")
debug.log.indent_level += 1
for letter in sorted(per_letter_results.keys()):
    outcomes = per_letter_results[letter]
    letter_avg = sum(outcomes) / len(outcomes) if outcomes else 0.0
    letter_delta = letter_avg - baseline_accuracy
    debug.log.analysis(f"Letter '{letter}' accuracy: {letter_avg:.2%}")
    debug.log.analysis(f"Letter '{letter}' delta from baseline: {letter_delta:.2%}")

debug.log.indent_level -= 1
