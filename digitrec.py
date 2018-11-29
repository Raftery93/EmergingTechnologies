# import required modules
import keras
import sys
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# import MNIST dataset
from keras.datasets import mnist

# load data
(x_train,y_train), (x_test,y_test) = mnist.load_data()

# preprocessing
x_test = x_test.reshape(x_test.shape[0], 784)
x_train = x_train.reshape(x_train.shape[0], 784)
x_test = x_test.astype('float32')
x_train = x_train.astype('float32')

x_test /= 255
x_train /= 255
y_test = keras.utils.to_categorical(y_test, 10)
y_train = keras.utils.to_categorical(y_train, 10)