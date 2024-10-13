import os
from mnist import mnist
import numpy as np

def prepare_data():
    current_working_directory = os.getcwd()
    if not os.path.exists(current_working_directory):
        os.makedirs(current_working_directory)
    current_working_directory = os.path.abspath(os.path.join(current_working_directory, ".."))
    
    images_training, labels_training, images_test, labels_test = mnist(f'{current_working_directory}/data')
    reshaped_images_training = [image.flatten().reshape(784, 1) for image in images_training]
    reshaped_labels_training = [np.eye(1, 10, lb).reshape(10, 1) for lb in labels_training]
    images_training = images_training/255
    images_test = images_test/255
    training_data = list(zip(reshaped_images_training, reshaped_labels_training))
    test_data = list(zip(images_test, labels_test))
    return training_data, test_data