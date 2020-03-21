from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from skimage.filters import gaussian
from matplotlib import image
import matplotlib.pyplot as plt
import keras
from keras.preprocessing.image import ImageDataGenerator, load_img
import numpy as np
import os
import sys
print( 'Using Keras version', keras.__version__)


# All images will  flipped, rotated up to 20º, and  shifted
train_datagen = ImageDataGenerator(horizontal_flip=True, 
                                   rotation_range=20, 
                                   width_shift_range=0.07,
                                   height_shift_range=0.07,)
test_datagen = ImageDataGenerator(horizontal_flip=True, 
                                  rotation_range=20, 
                                  width_shift_range=0.07,
                                  height_shift_range=0.07,)
# classes = ['Apple A', 'Apple C', 'Apple D', 'Apple E', 'Apple F', 'Banana', 'Carambola', 
#           'Guava A', 'Guava B', 'Kiwi A', 'Kiwi B', 'Kiwi C', 'Mango', 'Muskmelon', 
#           'Orange', 'Peach', 'Pear', 'Persimmon', 'Pitaya', 'Plum', 'Pomegranate', 'Tomatoes']
classes = os.listdir('fruit-recognition_reduced/')
classes = sorted(classes)


X, y = [], []
c = 0  # class id
for filename in classes: 
	# load image
        path = 'fruit-recognition_reduced/' + filename
        list_images = os.listdir(path)[:]
        print(f'{filename}: {len(list_images)}')
        for img in list_images:
                img_data = image.imread('fruit-recognition_reduced/' + filename + '/' + img)
                # Scale images to the range [0,1]
                img_data = img_data.astype(np.float32)/255.0
                # Remove noise
                img_data = gaussian(img_data, sigma=1, multichannel=True)
                # store loaded image
                X.append(img_data)
                y.append(c)
        c += 1
X = np.array(X)
y = np.array(y)
print(set(y))

# Compute class weights
from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(y),
                                                 y)
class_weights = dict(enumerate(class_weights))
print(class_weights)

# Split Train, Test and Validation sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.10, shuffle=True)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.10, shuffle=True)
print('Training size:', len(x_train), len(y_train))
print('Test size:', len(x_test), len(y_test))
print('Validation size:', len(x_val), len(y_val))


#Check sizes of dataset
# print('Number of train examples', x_train.shape[0])
# print('Size of train examples', x_train.shape[1:])


# Convert data types
x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)
x_val = x_val.astype(np.float32)

#Adapt the labels to the one-hot vector syntax required by the softmax
from keras.utils import np_utils
num_classes = len(set(y))
print(f'num classes: {num_classes}')
# y_train = np_utils.to_categorical(y_train, num_classes)
# y_test = np_utils.to_categorical(y_test, num_classes)
# y_val = np_utils.to_categorical(y_val, num_classes)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)


# Images resolution
img_rows, img_cols, channels = 77, 96, 3
input_shape = (img_rows, img_cols, channels)
#Reshape for input
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, channels)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, channels)
x_val = x_val.reshape(x_val.shape[0], img_rows, img_cols, channels)

#Define the CNN architecture
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout, SpatialDropout2D

# Model architecture
# (Conv-Pool)* layers
model = Sequential()
model.add(Conv2D(128, (5, 5), activation='selu', input_shape=input_shape))
model.add(SpatialDropout2D(0.2))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (5, 5), activation='selu'))
model.add(SpatialDropout2D(0.2))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='selu'))
model.add(SpatialDropout2D(0.2))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='selu'))
model.add(SpatialDropout2D(0.2))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(16, (3, 3), activation='selu'))
# model.add(SpatialDropout2D(0.2))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# Fully-conected layers
model.add(Flatten())
model.add(Dense(264, activation='selu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='selu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='selu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='selu'))
model.add(Dense(num_classes, activation='softmax'))

#Model visualization
#We can plot the model by using the ```plot_model``` function. We need to install *pydot, graphviz and pydot-ng*.
#from keras.util import plot_model
#plot_model(model, to_file='model.png', show_shapes=true)

#Compile the CNN
adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False, decay=1e-3)
model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])


#Start training
# history = model.fit(x_train,y_train,batch_size=64,epochs=5, validation_split=0.2)
# history = model.fit_generator(x_train, y_train, batch_size=64,epochs=5, validation_split=0.2)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,
                   patience=15, restore_best_weights=True)
batch_size = 32
history = model.fit_generator(train_datagen.flow(x_train, y_train, batch_size=batch_size), 
                              validation_data=test_datagen.flow(x_val, y_val, batch_size=batch_size),
                              validation_steps=len(x_val)/batch_size,
                              steps_per_epoch=len(x_train) / batch_size,
                              epochs=1000,
                              class_weight=class_weights,
                              workers=2,
                              callbacks=[es])

# Results and visualization
import matplotlib
matplotlib.use('Agg')

#Evaluate the model with test set
score = model.evaluate(x_test, y_test, verbose=1)
print('test loss:', score[0])
print('test accuracy:', score[1])

##Store Plots
#Accuracy plot
print(history.history.keys())
# Depending on the tensorflow version the key changes
if 'accuracy' in history.history:
        key = 'accuracy'
else:
        key = 'acc'
plt.plot(history.history[key], label='train acc')
plt.plot(history.history[f'val_{key}'], label='validation acc')
#Loss plot
plt.plot(history.history['loss'], label='train loss' )
plt.plot(history.history['val_loss'], label='validation loss')
plt.title('model loss/accuracy')
plt.ylabel('loss/accuracy')
plt.xlabel('epoch')
plt.legend(loc='upper right')
plt.savefig('cnn_acc_loss.pdf')

#Confusion Matrix
from sklearn.metrics import classification_report, confusion_matrix
#Compute probabilities
Y_pred = model.predict(x_test)
#Assign most probable label
y_pred = np.argmax(Y_pred, axis=1)
#Plot statistics
print( 'Analysis of results' )
print(classes)
print(classification_report(np.argmax(y_test,axis=1), y_pred, target_names=classes))
print(confusion_matrix(np.argmax(y_test,axis=1), y_pred))
plt.close()

# Plot confussion matrix
import seaborn as sns
fig, ax = plt.subplots(1, figsize=(12,12))
cm = confusion_matrix(np.argmax(y_test,axis=1), y_pred)
sns.heatmap(cm, annot=True, ax = ax, fmt="d")  #annot=True to annotate cells
# labels, title and ticks of the confussion matrix
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(classes, rotation=45)
ax.yaxis.set_ticklabels(classes, rotation=0)
plt.savefig('confussion_matrix.pdf')


#Saving model and weights
model_json = model.to_json()
with open('model.json', 'w') as json_file:
        json_file.write(model_json)
weights_file = "weights-MNIST_"+str(round(score[1], 4))+".hdf5"
model.save_weights(weights_file,overwrite=True)
os.mkdir(f'exectution_{round(score[1], 4)}')

#Loading model and weights
# from keras.models import model_from_json

#json_file = open('model.json','r')
#model_json = json_file.read()
#json_file.close()
#model = model_from_json(model_json)
#model.load_weights(weights_file)
