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

# create model
model = Sequential()
model.add(Dense(16, input_dim=784, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(10, activation='softmax'))

# compile the model
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

# fit model to the training data a.k.a. "training"
model.fit(x_train, y_train, batch_size=100, epochs=10, verbose=1)

# evaluate the model on test set 
score = model.evaluate(x_test, y_test, batch_size=100, verbose=0)

print("\nTest set loss: ", score[0])
print("Test set accuracy: ", score[1])