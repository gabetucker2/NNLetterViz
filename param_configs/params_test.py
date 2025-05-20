# import scripts
import functions.learning_funcs as learning_funcs
import functions.activation_funcs as activation_funcs
import functions.noise_funcs as noise_funcs

# epochs
num_epochs = 100

# training/testing data
train_test_split_ratio = 0.93

# NN learning params
μ = 0.01
num_hidden_layers = 2
num_neurons_per_hidden_layer = 64

# NN algorithms
learning_algorithm = learning_funcs.widrow_hoff_learning
learning_algorithm_deep = learning_funcs.widrow_hoff_learning_deep
activation_function = activation_funcs.activation_function_linear

# NN noise
axon_pot_interference = 0
noise_function = noise_funcs.normal_noise

# NN axon conductances
init_axon_mean = 0
init_axon_sd = 0.6

axon_weight_max_dev = 10

# misc
max_print_repeats = 99999
enable_visuals = False
