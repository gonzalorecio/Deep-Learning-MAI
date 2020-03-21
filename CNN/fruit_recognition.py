
from os import listdir
from os.path import isdir, join
from skimage import io
import numpy as np
import random

def obtain_data(category, path):
    items = listdir(path)
    x, y = [], []
    for item in items:
        if isdir(join(path, item)):
            xtemp, ytemp = obtain_data(str(item), path+"/"+str(item))
            x.extend(xtemp)
            y.extend(ytemp)
        else:
            x.append(join(path, item))
            y.append(category)
    return x, y


def load_images(X):
    return np.array([np.array(io.imread(x)) for x in X])
    #images = []
    #for i in range(len(X)):
    #    img = io.imread(X[i])
    #    images.append(img)
    #return images


# Function load_data: loads the train and test data and returns it as a pair of tuples.
# max_num specifies the maximum number of instances to use from the entire dataset.
# p specifies the percentage split of the instances to be used for the training data.
# Input: int max_num, float p
# Output: (x_train, y_train), (x_test, y_test)
def load_data(train_split = 0.7):
    X, Y = obtain_data("root", "fruit-recognition_reduced")
    data = list(zip(X, Y))
    random.shuffle(data)

    num_train = int(len(data) * train_split)
    train = data[:num_train]
    test = data[num_train:]
    x_train, y_train = zip(*train)
    x_test, y_test = zip(*test)

    x_train = load_images(x_train)
    x_test = load_images(x_test)
    return (x_train, y_train), (x_test, y_test)



# (x_train, y_train), (x_test, y_test) = cifar10.load_data()