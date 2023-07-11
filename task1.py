
import numpy as np
import time

# CPU Matrix Multiplication
def cpu_matrix_multiplication(matrix_a, matrix_b):
    start_time = time.time()

    # Perform matrix multiplication using CPU
    cpu_result = np.dot(matrix_a, matrix_b)

    end_time = time.time()
    execution_time = end_time - start_time
    return cpu_result, execution_time

# GPU Matrix Multiplication
def gpu_matrix_multiplication(matrix_a, matrix_b):
    start_time = time.time()

    # Perform matrix multiplication using GPU
    gpu_matrix_a = np.asarray(matrix_a)
    gpu_matrix_b = np.asarray(matrix_b)
    gpu_result = np.dot(gpu_matrix_a, gpu_matrix_b)
    #gpu_result = np.asnumpy(gpu_result)

    end_time = time.time()
    execution_time = end_time - start_time
    return gpu_result, execution_time
