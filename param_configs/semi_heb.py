# import scripts
import functions.learning_funcs as learning_funcs
import functions.activation_funcs as activation_funcs
import functions.noise_funcs as noise_funcs

# epochs
num_epochs = 1000

# training/testing data
train_test_split_ratio = 0.94

# NN learning params
μ = 0.05
num_hidden_layers = 1
num_neurons_per_hidden_layer = 128

# NN algorithms
learning_algorithm = learning_funcs.semisup_norm_hebbian_learning
learning_algorithm_deep = learning_funcs.semisup_norm_hebbian_learning_deep
fwd_activation_function = activation_funcs.activation_linear
back_activation_function = activation_funcs.activation_linear_derivative

# NN noise
axon_pot_interference = 0
noise_function = noise_funcs.normal_noise

# NN axon conductances
init_axon_mean = 0
init_axon_sd = 0.6
axon_weight_max_dev = 9999

# misc
max_print_repeats = 999999
enable_visuals = False
