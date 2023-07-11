# ************************ Written by Alireza ************************************
import json
import cv2
from datetime import datetime
import numpy as np


import task1 as t1
import task2 as t2

def runTaskOne():
    # Define the matrix sizes
    matrix_size = 1000
    matrix_a = np.random.rand(matrix_size, matrix_size)
    matrix_b = np.random.rand(matrix_size, matrix_size)
    # Run CPU Matrix Multiplication
    cpu_result, cpu_execution_time = t1.cpu_matrix_multiplication(matrix_a, matrix_b)
    # Run GPU Matrix Multiplication
    gpu_result, gpu_execution_time = t1.gpu_matrix_multiplication(matrix_a, matrix_b)
    # Data to be written
    dictionary = {
        "CPU Result": str(cpu_result),
        "CPU Execution Time": str(cpu_execution_time) + " seconds",
        "GPU Result": str(gpu_result),
        "GPU Execution Time": str(gpu_execution_time) + " seconds",
    }
    # Serializing json
    json_object = json.dumps(dictionary, indent=4)
    # datetime now
    dt_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Writing to sample.json
    with open("json/Result_Task1_" + dt_now + ".json", "w") as outfile:
        outfile.write(json_object)
    outfile.close()

def runTaskTwo():
    # Load an image for processing
    image_path = "image.jpg"
    image = cv2.imread(image_path)
    # Run CPU Image Processing
    cpu_result, cpu_execution_time = t2.cpu_image_processing(image)

    # Run GPU Image Processing
    gpu_result, gpu_execution_time = t2.gpu_image_processing(image)

    #cv2.waitKey(0)

    dictionary = {
        "CPU Result": str(cv2.imshow(cpu_result)),
        "GPU Result": str(cv2.imshow(gpu_result)),
        "CPU Execution Time": str(cpu_execution_time) + "seconds",
        "GPU Execution Time": str(gpu_execution_time) + "seconds",
    }
    # Serializing json
    json_object = json.dumps(dictionary, indent=4)
    # datetime now
    dt_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Writing to sample.json
    with open("json/Result_Task2_" + dt_now + ".json", "w") as outfile:
        outfile.write(json_object)
    outfile.close()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    while True:
        taskNumber = input('Which task should be executed?( 1 or 2 for end: 0 ):')
        if taskNumber == str(1):
            runTaskOne()
        if taskNumber == str(2):
            runTaskTwo()
        if taskNumber == str(0):
            break
    print('Stop!')



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
