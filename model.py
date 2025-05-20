# script imports
import debug
import data.letter_data as letter_data
import param_configs.params_test as params
import functions.math_funcs as math_funcs
import functions.fwd_prop_funcs as fwd_prop_funcs
import functions.axon_funcs as axon_funcs
import render

# library imports
from collections import defaultdict
import numpy as np 

##################################################################
# Initialize

debug.log.indent_level = 0
debug.log.procedure("Setting up layers...")

I = len(math_funcs.matrix_to_vector(list(letter_data.letter_variants.values())[0][0]))
HN = params.num_neurons_per_hidden_layer
HM = params.num_hidden_layers
O = len(letter_data.letter_variants)

layer_sizes = [I] + [HN] * HM + [O]
if params.enable_visuals:
    render.initialize_network_canvas(layer_sizes)

##################################################################

debug.log.procedure("Beginning simulation...")

epoch_letter_accuracies = []
debug.log.indent_level += 1
for epoch in range(params.num_epochs):

    ##################################################################

    debug.log.epoch(f"Epoch {epoch + 1} / {params.num_epochs}")

    debug.log.epoch("Splitting and randomizing order of data for training and testing...")
    training_data, testing_data = math_funcs.per_class_shuffle_split(
        letter_data.letter_variants, params.train_test_split_ratio)

    flattened_training = [
        (label, matrix)
        for label, matrices in training_data.items()
        for matrix in matrices
    ]
    np.random.shuffle(flattened_training)

    debug.log.axons("Initializing axon conductances...")
    W_matrix = axon_funcs.get_axons(I, HN, HM, O)
    # debug.log.axons(f"Axon matrix before training:\n{W_matrix}")

    ##################################################################

    debug.log.training("Beginning training...")
    debug.log.indent_level += 1
    for i, (actual_letter, training_matrix) in enumerate(flattened_training):
        debug.log.training(f"Training on matrix of letter type '{actual_letter}'")

        X = math_funcs.matrix_to_vector(training_matrix)
        T = math_funcs.one_hot(actual_letter, O)
        W_matrix = params.learning_algorithm_deep(X, W_matrix, t=T)

    debug.log.indent_level -= 1

    # debug.log.training(f"Matrix after training:\n{W_matrix}")

    ##################################################################

    debug.log.testing("Beginning testing...")

    letter_accuracies = {}

    debug.log.indent_level += 1
    for actual_letter, testing_matrices in testing_data.items():
        debug.log.testing(f"Testing matrix of letter type '{actual_letter}'")

        debug.log.indent_level += 1
        for i, matrix in enumerate(testing_matrices):
            # debug.log.testing(f"Iterating over letter '{actual_letter}' instance {i}")
            X = math_funcs.matrix_to_vector(matrix)
            layer_outputs = fwd_prop_funcs.fwd_prop_deep(X, W_matrix, return_all_layers=True)
            y = layer_outputs[-1]
            predicted_index = np.argmax(y)
            predicted_letter = list(letter_data.letter_variants.keys())[predicted_index]

            is_correct = (predicted_letter == actual_letter)

            # debug.log.testing(f"Output membrane potentials: {y}")
            debug.log.testing(f"Predicted: '{predicted_letter}'; Actual: '{actual_letter}'")

            if actual_letter not in letter_accuracies:
                letter_accuracies[actual_letter] = []
            letter_accuracies[actual_letter].append(is_correct)

            if params.enable_visuals:
                render.update_activations(
                    layer_outputs,
                    actual_letter=actual_letter,
                    predicted_letter=predicted_letter,
                    output_vector=y
                )
                render.wait_for_click()

        debug.log.indent_level -= 1

    debug.log.indent_level -= 1

    ##################################################################

    epoch_letter_accuracies.append(letter_accuracies)

    correct = sum(sum(results) for results in letter_accuracies.values())
    total = sum(len(results) for results in letter_accuracies.values())
    epoch_accuracy = correct / total if total else 0.0
    debug.log.analysis(f"Model accuracy in this epoch: {epoch_accuracy:.2%}")

    ##################################################################

debug.log.indent_level -= 1

##################################################################

if params.enable_visuals:
    render.exec_app()

##################################################################

debug.log.analysis("Beginning analytics...")
debug.log.analysis("Computing average accuracies across all epochs and letters...")
total_correct = 0
total_instances = 0
per_letter_results = defaultdict(list)

for epoch_data in epoch_letter_accuracies:
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

debug.log.analysis(f"Number of epochs: {params.num_epochs}")
debug.log.analysis(f"Average model accuracy across {params.num_epochs} epochs: {average_accuracy:.2%}")
debug.log.analysis(
    f"Model accuracy delta: {accuracy_delta:.2%} "
    f"({average_accuracy:.2%} model accuracy vs {baseline_accuracy:.2%} random guess accuracy)")

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
