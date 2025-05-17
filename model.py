# script imports
import debug
import data.letterData as letterData
import paramConfigs.paramsTest as params
import functions.mathFuncs as mathFuncs
import functions.fwdPropFuncs as fwdPropFuncs
import functions.axonFuncs as axonFuncs
import render

# library imports
from collections import defaultdict
import numpy as np 
from PyQt5.QtTest import QTest

##################################################################

debug.log.indent_level = 0
debug.log.procedure(f"Setting up variables...")

I = len(mathFuncs.matrix_to_vector(list(letterData.letterVariants.values())[0][0]))
HN = params.numNeuronsPerHiddenLayer # neurons per hidden layer
HM = params.numHiddenLayers          # num hidden layers
O = len(letterData.letterVariants)   # fetch number of letters

layer_sizes = [I] + [HN] * HM + [O]
if params.enableVisuals:
    render.initialize_network_canvas(layer_sizes)

epochLetterAccuracies = []

##################################################################

debug.log.procedure("Beginning simulation...")
debug.log.indent_level += 1
for epoch in range(params.numEpochs):

    ##################################################################

    debug.log.epoch(f"Epoch {epoch+1} / {params.numEpochs}")

    debug.log.epoch("Splitting and randomizing order of data for training and testing...")
    trainingData, testingData = mathFuncs.per_class_shuffle_split(letterData.letterVariants, params.trainTestSplitRatio)

    flattened_training = [
        (label, matrix)
        for label, matrices in trainingData.items()
        for matrix in matrices
    ]
    np.random.shuffle(flattened_training)

    debug.log.axons("Initializing axon conductances...")
    W_matrix = axonFuncs.getAxons(I, HN, HM, O)
    # debug.log.axons(f"Axon matrix before training:\n{W_matrix}")

    ##################################################################

    debug.log.training(f"Beginning training...")
    debug.log.indent_level += 1
    for i, (actualLetter, trainingMatrix) in enumerate(flattened_training):
        debug.log.training(f"Training on matrix of letter type '{actualLetter}'")

        X = mathFuncs.matrix_to_vector(trainingMatrix)
        T = mathFuncs.one_hot(actualLetter, O)
        W_matrix = params.learningAlgorithmDeep(X, W_matrix, T=T)

    debug.log.indent_level -= 1

    # debug.log.training(f"Matrix after training:\n{W_matrix}")

    ##################################################################
    
    debug.log.testing(f"Beginning testing...")

    letterAccuracies = {}

    debug.log.indent_level += 1
    for actualLetter, testingMatrices in testingData.items():
        debug.log.testing(f"Testing matrix of letter type '{actualLetter}'")
        
        debug.log.indent_level += 1
        for i, matrix in enumerate(testingMatrices):
            # debug.log.testing(f"Iterating over letter '{actualLetter}' instance {i}")
            X = mathFuncs.matrix_to_vector(matrix)
            layer_outputs = fwdPropFuncs.fwdPropDeep(X, W_matrix, return_all_layers=True)
            y = layer_outputs[-1]
            predictedLetterIndex = np.argmax(y)
            predictedLetter = list(letterData.letterVariants.keys())[predictedLetterIndex]

            isCorrect = (predictedLetter == actualLetter)

            # debug.log.testing(f"Output membrane potentials: {y}")
            debug.log.testing(f"Predicted: '{predictedLetter}'; Actual: '{actualLetter}'")

            if actualLetter not in letterAccuracies:
                letterAccuracies[actualLetter] = []
            letterAccuracies[actualLetter].append(isCorrect)

            if params.enableVisuals:
                render.update_activations(
                    layer_outputs,
                    actual_letter=actualLetter,
                    predicted_letter=predictedLetter,
                    output_vector=y
                )
                QTest.qWait(500)

        debug.log.indent_level -= 1

    debug.log.indent_level -= 1

    ##################################################################

    epochLetterAccuracies.append(letterAccuracies)

    correct = 0
    total = 0
    for results in letterAccuracies.values():
        correct += sum(results)
        total += len(results)

    epochAccuracy = correct / total if total else 0.0
    debug.log.analysis(f"Model accuracy in this epoch: {epochAccuracy:.2%}")

    ##################################################################

debug.log.indent_level -= 1

##################################################################

debug.log.analysis("Beginning analytics...")

debug.log.analysis("Computing average accuracies across all epochs and letters...")
total_correct = 0
total_instances = 0
per_letter_results = defaultdict(list)

# Flatten epoch results to get per-letter performance across all epochs
for epoch_data in epochLetterAccuracies:
    if not isinstance(epoch_data, dict):
        debug.log.warning("Skipping invalid epoch data format (expected dict).")
        continue
    for letter, results in epoch_data.items():
        if not isinstance(results, list):
            debug.log.warning(f"Skipping results for letter '{letter}' (expected list).")
            continue
        per_letter_results[letter].extend(results)
        total_correct += sum(results)
        total_instances += len(results)

average_accuracy = total_correct / total_instances if total_instances else 0.0

debug.log.analysis("Computing baseline random guess accuracy...")
num_letters = len(per_letter_results)
baseline_accuracy = 1 / num_letters if num_letters else 0.0

debug.log.analysis("Computing accuracy delta...")
accuracy_delta = average_accuracy - baseline_accuracy

##################################################################

debug.log.analysis("Final simulation analytics...")

debug.log.analysis(f"Number of epochs: {params.numEpochs}")
debug.log.analysis(f"Average model accuracy across {params.numEpochs} epochs: {average_accuracy:.2%}")
debug.log.analysis(f"Model accuracy delta: {accuracy_delta:.2%} ({average_accuracy:.2%} model accuracy vs {baseline_accuracy:.2%} random guess accuracy)")

debug.log.analysis("Average accuracy per letter:")
debug.log.indent_level += 1
for letter in sorted(per_letter_results.keys()):
    outcomes = per_letter_results[letter]
    if not outcomes:
        debug.log.analysis(f"Letter '{letter}' has no recorded results.")
        continue
    letter_avg = sum(outcomes) / len(outcomes)
    letter_delta = letter_avg - baseline_accuracy
    debug.log.analysis(f"Letter '{letter}' accuracy: {letter_avg:.2%}")
    debug.log.analysis(f"Letter '{letter}' delta from baseline: {letter_delta:.2%}")
debug.log.indent_level -= 1

##################################################################

if params.enableVisuals:
    render.exec_app()
