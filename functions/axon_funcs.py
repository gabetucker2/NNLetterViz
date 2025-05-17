# import scripts
import param_configs.params_test as params

# import libraries
import random

def init_axon_val():
    return random.normalvariate(mu=params.init_axon_mean, sigma=params.init_axon_sd)

def get_axons(i, hn, hm, o):
    def new_layer(n_out, n_in):
        return [[init_axon_val() for _ in range(n_in)] for _ in range(n_out)]

    # Input -> First hidden layer
    axons_input_hidden = new_layer(hn, i)  # hn x i

    # Hidden -> Hidden (hm - 1 layers)
    axons_hidden_hidden = [new_layer(hn, hn) for _ in range(hm - 1)]  # hm-1 layers of hn x hn

    # Last hidden -> Output
    axons_hidden_output = new_layer(o, hn)  # o x hn

    return [axons_input_hidden] + axons_hidden_hidden + [axons_hidden_output]
