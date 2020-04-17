# from tensorflow.python.client import device_lib
from sklearn.utils import class_weight

import sys
import pandas
from sklearn.metrics import confusion_matrix, classification_report
from nltk.corpus import stopwords
import re
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding
# from keras.layers import LSTM
from keras.optimizers import RMSprop, SGD, Adam
from keras.utils import np_utils
from collections import Counter
import argparse
import time
import json
from sklearn.preprocessing import LabelEncoder
from keras.layers.recurrent import GRU, SimpleRNN, LSTM
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.layers.core import Dropout, SpatialDropout1D
from keras import regularizers
from keras.layers.wrappers import Bidirectional
from keras.initializers import he_normal
# from keras import backend as K
# print(K.tensorflow_backend._get_available_gpus())
# print(str(device_lib.list_local_devices()))

def headline_to_words(raw_text):
    """
    Only keeps ascii characters in the tweet and discards @words

    :param raw_tweet:
    :return:
    """
    letters_only = re.sub("[^a-zA-Z @]", " ", raw_text)
    words = letters_only.lower().split()
    # words = letters_only.split()

    stops = set(stopwords.words('english'))
    meaningful_words = [w for w in words if not re.match("^[@]", w) and (w not in stops)]

    # print(meaningful_words)
    return " ".join(meaningful_words)


# if __name__ == '__main__':
# parser = argparse.ArgumentParser()
# parser.add_argument('--verbose', help="Verbose output (enables Keras verbose output)",
#                     action='store_true', default=False)
# args = parser.parse_args()

verbose = 1 #if args.verbose else 0
impl = 2
start = time.ctime()
print("Starting:", time.ctime())

############################################
# Data

def read_dataset():
    PATH = 'News_Category_Dataset_v2.json'
    data = []
    with open(PATH, 'r', encoding='utf-8') as f:
        headlines = list(map(lambda x: x.strip(), f.readlines()))
        for line in headlines:
            try:
                data.append(json.loads(line))
            except:
                print(line)
    df = pandas.DataFrame(data)
    return df

def balance_dataset(df, threshold=5000):
    dfs = []
    grouped = df.groupby(by='category')
    for _, group in grouped:
        # group = groups.get_group(g)
        dfs.append(group.sample(frac=1, random_state=1).reset_index(
            drop=True)[:threshold])
    return pandas.concat(dfs, ignore_index=True)

# Tweet = pandas.read_csv("Airlines.csv")
#    Tweet = pandas.read_csv("Presidential.csv")
News = read_dataset()
News.category = News.category.map(
    lambda x: "WORLDPOST" if x == "THE WORLDPOST" else x)
News = balance_dataset(News, threshold=2000)
cates = News.groupby('category')
print("total categories:", cates.ngroups)
print(cates.size())
# Pre-process the tweet and store in a separate column
News['clean_text'] = list(map(lambda p: p[0]+' '+p[1], zip(News['headline'].apply(lambda x: headline_to_words(x)), News['short_description'].apply(lambda x: headline_to_words(x)))))
print(News.clean_text.values[:2])
# Convert sentiment to binary
# Tweet['sentiment'] = Tweet['twsentiment'].apply(lambda x: 0 if x == 'negative' else 1 if x == 'positive' else 2)

# Join all the words in review to build a corpus
all_text = ' '.join(News['clean_text'])
words = all_text.split()

print(all_text[:1000])
# Convert words to integers
counts = Counter(words)
print('Num of different words:', len(set(words)))
numwords = 50000  # Limit the number of words to use
# numwords = int(len(set(words))*0.7)
vocab = sorted(counts, key=counts.get, reverse=True)[:numwords]
vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}

tweet_ints = []
for each in News['clean_text']:
    tweet_ints.append([vocab_to_int[word]
                        for word in each.split() if word in vocab_to_int])

# Create a list of labels
label_encoder = LabelEncoder()
News['category_label'] = label_encoder.fit_transform(News['category'])
labels = np.array(News['category_label'])
num_classes = len(set(labels))

# Compute class weights
class_weights = class_weight.compute_class_weight('balanced',
                                                    np.unique(labels),
                                                    labels)
class_weights = dict(enumerate(class_weights))
print(class_weights)

# Find the number of tweets with zero length after the data pre-processing
tweet_len = Counter([len(x) for x in tweet_ints])
print("Zero-length reviews: {}".format(tweet_len[0]))
print("Maximum tweet length: {}".format(max(tweet_len)))

# Remove those tweets with zero length and its corresponding label
tweet_idx = [idx for idx, tweet in enumerate(tweet_ints) if len(tweet) > 0]
labels = labels[tweet_idx]
Tweet = News.iloc[tweet_idx]
tweet_ints = [tweet for tweet in tweet_ints if len(tweet) > 0]

seq_len = max(tweet_len)
features = np.zeros((len(tweet_ints), seq_len), dtype=int)

left_padding = True

for i, row in enumerate(tweet_ints):
    if left_padding:
        features[i, -1*len(row):] = np.array(row)[:seq_len]
    else:
        features[i, :len(row)] = np.array(row)[:seq_len]

print(features[:3, :])
split_frac = 0.8
split_idx = int(len(features) * 0.8)
# train_x, val_x = features[:split_idx], features[split_idx:]
# train_y, val_y = labels[:split_idx], labels[split_idx:]

train_x, val_x, train_y, val_y = train_test_split(
    features, labels, test_size=0.2, random_state=42, shuffle=True)

test_idx = int(len(val_x) * 0.5)
# val_x, test_x = val_x[:test_idx], val_x[test_idx:]
# val_y, test_y = val_y[:test_idx], val_y[test_idx:]
val_x, test_x, val_y, test_y = train_test_split(
    val_x, val_y, test_size=0.5, random_state=42, shuffle=True)

print("\t\t\tFeature Shapes:")
print("Train set: \t\t{}".format(train_x.shape),
        "\nValidation set: \t{}".format(val_x.shape),
        "\nTest set: \t\t{}".format(test_x.shape))

print("Train set: \t\t{}".format(train_y.shape),
        "\nValidation set: \t{}".format(val_y.shape),
        "\nTest set: \t\t{}".format(test_y.shape))

############################################
batch_size = 256
# Model
drop = 0.15
rdrop = 0.0
l2_regularizer_embeddings = 0.015  # 0.005 # 
l2_regularizer = 0.0 # 0.0075 #0.

nlayers = 1  # >= 1
# RNN = LSTM  # GRU
# RNN = GRU
for i, RNN in enumerate([GRU]):
    name = ['GRU'][i]
    print(name, 'Bidirectional')
    neurons = 128 # 32  # seq_len  # 64
    embedding = 8  # 20

    model = Sequential()
    # model.add(Dropout(0.2))
    model.add(Embedding(numwords + 1, embedding, input_length=seq_len,
                        mask_zero=True, 
                        embeddings_regularizer=regularizers.l2(l2_regularizer_embeddings),
                        # embeddings_initializer=he_normal(seed=42)
                        ))
    model.add(Dropout(0.2))
    # model.add(SpatialDropout1D(0.2))
    if nlayers == 1:
        model.add(Bidirectional(RNN(neurons, implementation=impl, recurrent_dropout=rdrop,
                    dropout=drop, kernel_regularizer=regularizers.l2(l2_regularizer)
                    )))
    else:
        model.add(Bidirectional(RNN(neurons, implementation=impl,
                                    recurrent_dropout=rdrop, dropout=drop, return_sequences=True, 
                                    kernel_regularizer=regularizers.l2(l2_regularizer))))
        for i in range(1, nlayers - 1):
            model.add(Bidirectional(RNN(neurons, recurrent_dropout=rdrop, dropout=drop,
                                        implementation=impl, return_sequences=True, kernel_regularizer=regularizers.l2(l2_regularizer))))
        model.add(Bidirectional(RNN(neurons, recurrent_dropout=drop, dropout=drop,
                                    implementation=impl, kernel_regularizer=regularizers.l2(l2_regularizer))))
    # model.add(Dense(32, activation='selu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))

    ############################################
    # Training
    learning_rate = 0.01  # 0.025
    # learning_rate = [0.005, 0.01,0.001][i]
    optimizer = SGD(lr=learning_rate, momentum=0.95)
    optimizer = Adam(lr=learning_rate, beta_1=0.9,
                    beta_2=0.999, amsgrad=False, decay=1e-4)
    model.compile(loss='categorical_crossentropy',
                optimizer=optimizer, metrics=['accuracy'])

    epochs = 200
    # batch_size = 64

    train_y_c = np_utils.to_categorical(train_y, num_classes)
    val_y_c = np_utils.to_categorical(val_y, num_classes)

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,
                    patience=5, restore_best_weights=True)
    history = model.fit(train_x, train_y_c,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(val_x, val_y_c),
                        callbacks=[es],
                        # workers=2,
                        # class_weight=class_weights,
                        #   verbose=verbose,
                        )

    ############################################
    # Results

    test_y_c = np_utils.to_categorical(test_y, num_classes)
    score, acc = model.evaluate(test_x, test_y_c,
                                batch_size=batch_size,)
    # verbose=verbose)
    print(name, learning_rate, left_padding)
    print('Test ACC=', acc)
    from sklearn.metrics import f1_score

    test_pred = model.predict_classes(test_x, verbose=verbose)
    print('F1 score=', f1_score(test_y,test_pred, average='weighted'))
    np.set_printoptions(threshold=sys.maxsize)
    print(history.history)
    print('val_loss:', history.history['val_loss'])

    print()

    print()
    print('Confusion Matrix')
    print('-'*20)
    print(confusion_matrix(test_y, test_pred))
    print()
    print('Classification Report')
    print('-'*40)
    print(classification_report(test_y, test_pred))
    print()
    print("Ending:", time.ctime())

    import matplotlib.pyplot as plt

    if 'accuracy' in history.history:
        key = 'accuracy'
    else:
        key = 'acc'
    plt.plot(history.history[key], label='train acc')
    plt.plot(history.history[f'val_{key}'], label='validation acc')
    # Loss plot
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='validation loss')
    plt.title('model loss/accuracy')
    plt.ylabel('loss/accuracy')
    plt.xlabel('epoch')
    plt.legend(loc='upper right')
    plt.savefig(f'{name}_{left_padding}_acc_loss2.pdf')
    plt.show()
