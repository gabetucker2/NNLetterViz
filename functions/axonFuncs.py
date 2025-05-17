# import scripts
import paramConfigs.paramsTest as params

# import libraries
import random

def initAxonVal():
    return random.normalvariate(mu=params.initAxonMean, sigma=params.initAxonSD)

def getAxons(I, HN, HM, O):
    def new_layer(n_out, n_in):
        return [[initAxonVal() for _ in range(n_in)] for _ in range(n_out)]

    # Input -> First hidden layer
    axons_input_hidden = new_layer(HN, I)  # HN x I

    # Hidden -> Hidden (HM - 1 layers)
    axons_hidden_hidden = [new_layer(HN, HN) for _ in range(HM - 1)]  # HM-1 layers of HN x HN

    # Last hidden -> Output
    axons_hidden_output = new_layer(O, HN)  # O x HN

    return [axons_input_hidden] + axons_hidden_hidden + [axons_hidden_output]
