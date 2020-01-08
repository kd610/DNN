import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from utils import *

train_x_orig, train_y, test_x_orig, test_y, classes = load_dataset()

# Reshape the training and test examples 
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.

print ("train_x's shape: " + str(train_x.shape))
print ("test_x's shape: " + str(test_x.shape))

n_x = 12288     # num_px * num_px * 3
n_h = 7
n_y = 1
layers_dims = [12288, 20, 7, 5, 1]


parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations = 2500, print_cost = True)

#Show accuracy for train data
pred_train = predict(train_x, train_y, parameters)

#show accuracy for dev data
pred_test = predict(test_x, test_y, parameters)
