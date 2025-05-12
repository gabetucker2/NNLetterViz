# import scripts
import debug
import functions.axonFuncs as axonFuncs

# math functions
def create_matrix(n_rows, n_cols, init_function):
    return [[init_function() for _ in range(n_cols)] for _ in range(n_rows)]

def matrix_to_vector(matrix):
    if not isinstance(matrix, list):
        return [matrix]
    
    flattened_vector = []
    for element in matrix:
        flattened_vector.extend(flattenMatrix(element))

def vector_to_matrix(matrix):
    if not isinstance(matrix, list):
        return [matrix]
    
    flattened_vector = []
    for element in matrix:
        flattened_vector.extend(flattenMatrix(element))

    return flattened_vector

def mean(array):
    return sum(array) / len(array)

def sample_variance(sample):
    if len(sample) < 2:
        debug.Log.print_warning("Variance is not defined for a sample with fewer than 2 elements")
        return 0
    
    return sum((x - mean(sample)) ** 2 for x in sample) / (len(sample) - 1)
