# import scripts
import debug

# import libraries
import math
import random

# math functions
def matrix_to_vector(matrix):   
    flattened_vector = []
    for element in matrix:
        flattened_vector.extend(element)

    return flattened_vector

def vector_to_matrix(vector):
    length = len(vector)
    root = int(math.sqrt(length))
    if root * root != length:
        raise ValueError("Length of vector is not a perfect square")

    matrix = []
    for i in range(root):
        row = vector[i * root : (i + 1) * root]
        matrix.append(row)

    return matrix

def mean(array):
    return sum(array) / len(array)

def sample_variance(sample):
    if len(sample) < 2:
        debug.Log.print_warning("Variance is not defined for a sample with fewer than 2 elements")
        return 0
    
    return sum((x - mean(sample)) ** 2 for x in sample) / (len(sample) - 1)

def dot_product(v1, v2):
    sum = 0
    for i in enumerate(v1):
        sum += v1[i]*v2[i]
    return sum

def shuffle_split(variant_dict, splitRatio):
    trainingData = {}
    testingData = {}
    for letter, matrices in variant_dict.items():
        n = len(matrices)
        num_train = math.ceil(splitRatio * n)
        shuffled = matrices[:]
        random.shuffle(shuffled)
        trainingData[letter] = shuffled[:num_train]
        testingData[letter] = shuffled[num_train:]
    return trainingData, testingData

def compute_letter_acc(letter, letterAccuracies):
    results = letterAccuracies.get(letter, [])
    if not results:
        debug.Log.warning("Unable to compute get accuracies")
        return 0.0
    
    return sum(results) / len(results)
