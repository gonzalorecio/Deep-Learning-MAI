from __future__ import division
import keras
import tensorflow as tf
# from keras.datasets import fruit_recognition as fruit
import fruit_recognition as fruit
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten
import time
from sklearn import metrics
import numpy as np


print("Begin Session")
time_init = time.time()

# Load Data
(x_train, y_train), (x_test, y_test_original) = fruit.load_data(train_split=0.8)

y_test = y_test_original

y_values = list(set(y_train + y_test))

print("Number of Train instances: ", len(x_train))
print("Number of Test instances: ", len(x_test))
print("Number of Class Values: ", len(y_values))
for v in y_values:
    print("Class ", v, ": Train->",list(y_train).count(v)," Test->",list(y_test).count(v))

input_shape = x_train[0].shape
print("Image shape: ", input_shape)

#Normalize data
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train / 255
x_test = x_test / 255

#Adapt the labels to the one-hot vector syntax required by the softmax
y_train = [y_values.index(val) for val in y_train]
y_test_int = [y_values.index(val) for val in y_test]
y_train = np_utils.to_categorical(y_train, len(y_values))
y_test = np_utils.to_categorical(y_test_int, len(y_values))


# Builf the Model
model = Sequential()
model.add(Conv2D(8, 4, 4, activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Conv2D(2, 2, 1, activation='relu', input_shape=input_shape))
# model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
# model.add(Dense(10, activation='relu'))
model.add(Dense(len(y_values), activation=(tf.nn.softmax)))

model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])
history = model.fit(x_train,y_train,epochs=10)

score = model.evaluate(x_test, y_test, verbose=0)
y_pred = np.array(model.predict(x_test))
y_pred = np.argmax(y_pred, axis=1)
conf_matrix = metrics.confusion_matrix(y_test_int, y_pred)

y_values = [(i, y_values[i]) for i in range(len(y_values))]
print("Classes: ", y_values)
print("\nConfusion Matrix:\n")
for row in conf_matrix:
    print(list(row))

print('test loss:', score[0])
print('test accuracy:', score[1])

print("\nPrecision score:\n", metrics.precision_score(y_test_int, y_pred, average=None))
print("\nRecall score:\n", metrics.recall_score(y_test_int, y_pred, average=None))
print("\nF1 score:\n", metrics.f1_score(y_test_int, y_pred, average=None))
print("\nClassification report:\n", metrics.classification_report(y_test_int, y_pred))

print("Total Time: ", (time.time()-time_init))