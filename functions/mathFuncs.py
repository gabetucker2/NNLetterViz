# import scripts
import debug
import data.letterData as letterData

# import libraries
import math
import random
from collections import defaultdict
import numpy as np

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

def dot_product(v1, v2):
    sum = 0
    for i in enumerate(v1):
        sum += v1[i]*v2[i]
    return sum

def shuffle_split(letterVariants, ratio):
    all_data = []

    # Flatten all letter samples into (label, matrix) pairs
    for letter, matrices in letterVariants.items():
        for matrix in matrices:
            all_data.append((letter, matrix))

    # Shuffle the full list
    random.shuffle(all_data)

    # Split
    split_idx = int(len(all_data) * ratio)
    train_raw = all_data[:split_idx]
    test_raw = all_data[split_idx:]

    # Group back by letter
    train = defaultdict(list)
    test = defaultdict(list)

    for label, matrix in train_raw:
        train[label].append(matrix)
    for label, matrix in test_raw:
        test[label].append(matrix)

    return train, test

def per_class_shuffle_split(letterVariants, train_ratio):
    from random import shuffle

    train_split = {}
    test_split = {}

    for letter, matrices in letterVariants.items():
        shuffled = matrices.copy()
        shuffle(shuffled)
        split_index = int(len(shuffled) * train_ratio)
        train_split[letter] = shuffled[:split_index]
        test_split[letter] = shuffled[split_index:]

    return train_split, test_split

def one_hot(letter, num_classes):
    vec = np.zeros(num_classes)
    index = list(letterData.letterVariants.keys()).index(letter)
    vec[index] = 1.0
    return vec
