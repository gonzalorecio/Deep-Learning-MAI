from keras.callbacks import EarlyStopping
from keras import applications
from sklearn.model_selection import train_test_split
from skimage.filters import gaussian
from matplotlib import image
from keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import keras
from keras.preprocessing.image import ImageDataGenerator, load_img
import numpy as np
import os
import sys
print( 'Using Keras version', keras.__version__)
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import preprocess_input as preprocess_vgg
from keras.applications.resnet50 import preprocess_input as preprocess_resnet
from keras.applications.inception_v3 import preprocess_input as preprocess_inception
from PIL import Image


# All images will  flipped, rotated up to 20º, and  shifted up to 7% of the height and width
train_datagen = ImageDataGenerator(
                                # horizontal_flip=True, 
                                #    rotation_range=20, 
                                #    width_shift_range=0.07,
                                #    height_shift_range=0.07,
                                   )
test_datagen = ImageDataGenerator(
                                # horizontal_flip=True, 
                                #   rotation_range=20, 
                                #   width_shift_range=0.07,
                                #   height_shift_range=0.07,
                                  )
# classes = ['Apple A', 'Apple C', 'Apple D', 'Apple E', 'Apple F', 'Banana', 'Carambola', 
#           'Guava A', 'Guava B', 'Kiwi A', 'Kiwi B', 'Kiwi C', 'Mango', 'Muskmelon', 
#           'Orange', 'Peach', 'Pear', 'Persimmon', 'Pitaya', 'Plum', 'Pomegranate', 'Tomatoes']
ROOT_DIR = '../datasets/'
classes = os.listdir(ROOT_DIR + 'fruit-recognition_reduced/')
classes = sorted(classes)


X, y = [], []
c = 0  # class id
for filename in classes: 
	# load image
        path = ROOT_DIR + 'fruit-recognition_reduced/' + filename
        list_images = os.listdir(path)[:100]
        print(f'{filename}: {len(list_images)}')
        for img in list_images:
                # img_data = image.imread(ROOT_DIR + 'fruit-recognition_reduced/' + filename + '/' + img)
                img_data = load_img(ROOT_DIR + 'fruit-recognition_reduced/' + filename + '/' + img, target_size=(224,224))
                # img_data = img_data.resize((224, 224), Image.ANTIALIAS)
                img_data = img_to_array(img_data)
                # Scale images to the range [0,1]
                # img_data = img_data.astype(np.float32)/255.0
                # img_data = img_data.astype(np.float32) - [123.68, 116.779, 103.939]
                
                # img_data = preprocess_vgg(img_data.astype(np.float32))
                img_data = preprocess_inception(img_data.astype(np.float32))
                # img_data = preprocess_resnet(img_data.astype(np.float32))

                # Remove noise
                # img_data = gaussian(img_data, sigma=1, multichannel=True)
                # store loaded image
                X.append(img_data)
                y.append(c)
        c += 1
X = np.array(X)
y = np.array(y)
print(set(y))

# Compute class weights
from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight('balanced', np.unique(y), y)
class_weights = dict(enumerate(class_weights))
print(class_weights)

# Split Train, Test and Validation sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.10, shuffle=True, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.10, shuffle=True, random_state=42)
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
# img_rows, img_cols, channels = 77, 96, 3
img_rows, img_cols, channels = 224, 224, 3
input_shape = (img_rows, img_cols, channels)
#Reshape for input
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, channels)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, channels)
x_val = x_val.reshape(x_val.shape[0], img_rows, img_cols, channels)

#Define the CNN architecture
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout, SpatialDropout2D
# new_input = Input(shape=input_shape)
# model_trained = keras.applications.VGG16(
model_trained = InceptionV3(
# model_trained = ResNet50(
#     weights=None, 
    include_top=False, 
    input_shape=input_shape)

# model = model_trained
# Freeze the layers which you don't want to train. Here I am freezing the first 5 layers.
print('Number of layers in the trained model:', len(model_trained.layers))
half = int(len(model_trained.layers)/2)
for layer in model_trained.layers[:]:
    layer.trainable = False


x = model_trained.output
features = Flatten()(x)
model = Model(inputs=model_trained.input, output=features)


#Compile the CNN
adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False, decay=1e-3)
sgd = keras.optimizers.SGD(lr=0.0001, momentum=0.9)
model.compile(optimizer=sgd, loss='categorical_crossentropy',metrics=['accuracy'])
print(model.summary())

#Start training
# history = model.fit(x_train,y_train,batch_size=64,epochs=5, validation_split=0.2)
# history = model.fit_generator(x_train, y_train, batch_size=64,epochs=5, validation_split=0.2)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,
                   patience=10, restore_best_weights=True)
batch_size = 32
# history = model.fit_generator(train_datagen.flow(x_train, y_train, batch_size=batch_size), 
#                               validation_data=test_datagen.flow(x_val, y_val, batch_size=batch_size),
#                               validation_steps=len(x_val)/batch_size,
#                               steps_per_epoch=len(x_train) / batch_size,
#                               epochs=1000,
#                         #       class_weight=class_weights,
#                               workers=2,
#                               callbacks=[es])

from sklearn.metrics import classification_report, confusion_matrix
from sklearn import svm
import time


#Train SVM with the obtained features.
clf = svm.LinearSVC()
train_feats = model.predict(x_train)
test_feats = model.predict(x_test)
print(train_feats.shape)
print(np.argmax(y_train, axis=1).shape)
print(y_train.shape)

start = time.time()
clf.fit(X = train_feats, y = np.argmax(y_train, axis=1))
end = time.time()
print("SVM training time (s):", end - start)
print('Done training SVM on extracted features of training set')


#Test SVM with the test set.
predicted_labels = clf.predict(test_feats)
print('Done testing SVM on extracted features of test set')

#Print results
print(classification_report(np.argmax(y_test,axis=1), predicted_labels))
print(confusion_matrix(np.argmax(y_test,axis=1), predicted_labels))
from sklearn.metrics import f1_score, accuracy_score
print('f1 score macro', f1_score(np.argmax(y_test, axis=1), predicted_labels, average='macro'))
print('accuracy score', accuracy_score(np.argmax(y_test, axis=1), predicted_labels))


# Results and visualization
import matplotlib
matplotlib.use('Agg')

#Evaluate the model with test set
# score = model.evaluate(x_test, y_test, verbose=1)
# print('test loss:', score[0])
# print('test accuracy:', score[1])

# ##Store Plots
# #Accuracy plot
# print(history.history.keys())
# # Depending on the tensorflow version the key changes
# if 'accuracy' in history.history:
#         key = 'accuracy'
# else:
#         key = 'acc'
# plt.plot(history.history[key], label='train acc')
# plt.plot(history.history[f'val_{key}'], label='validation acc')
# #Loss plot
# plt.plot(history.history['loss'], label='train loss' )
# plt.plot(history.history['val_loss'], label='validation loss')
# plt.title('model loss/accuracy')
# plt.ylabel('loss/accuracy')
# plt.xlabel('epoch')
# plt.legend(loc='upper right')
# plt.savefig('cnn_acc_loss.pdf')




# Plot confussion matrix
import seaborn as sns
fig, ax = plt.subplots(1, figsize=(12,12))
cm = confusion_matrix(np.argmax(y_test,axis=1), predicted_labels)
sns.heatmap(cm, annot=True, ax = ax, cmap="Blues", fmt="d")  #annot=True to annotate cells
# labels, title and ticks of the confussion matrix
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(classes, rotation=45)
ax.yaxis.set_ticklabels(classes, rotation=0)
plt.savefig('confussion_matrix_SVM.pdf')




#Loading model and weights
# from keras.models import model_from_json

#json_file = open('model.json','r')
#model_json = json_file.read()
#json_file.close()
#model = model_from_json(model_json)
#model.load_weights(weights_file)
