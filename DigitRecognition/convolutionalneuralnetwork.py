import sys
from io import StringIO

import time
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as kb
from keras.models import load_model

kb.set_image_dim_ordering('th')
pixel_max_val = 255


def train_model():
    # import (download) data from MNIST
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # reshape to be [samples][pixels][width][height]
    X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

    # normalize inputs from 0-255 to 0-1
    X_train /= 255
    X_test /= 255

    # one hot encode outputs
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    num_classes = y_test.shape[1]

    # build CNN model
    def larger_model():
        # create model
        model = Sequential()
        model.add(Conv2D(30, (5, 5), input_shape=(1, 28, 28), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(15, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    # build the model
    model = larger_model()

    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    start = time.time()

    # Fit the model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200)

    end = time.time()
    sys.stdout = old_stdout

    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=2)
    print("Large CNN Error: %.2f%%" % (100 - scores[1] * 100))

    stats = mystdout.getvalue().splitlines()[20].split(' ')
    scores = model.evaluate(X_test, y_test, verbose=0)

    cnn_stats = ["Accuracy: {}\n".format(stats[6]),
                 "Value accuracy: {}\n".format(stats[12]),
                 "Loss: {}\n".format(stats[3]),
                 "Value loss: {}\n".format(stats[9]),
                 "Time to train: {}\n".format(end - start)]

    print('Baseline Error: %.2f%%' % (100 - scores[1] * 100))

    return model, cnn_stats


def load(path):
    return load_model(path)


def save(path, model):
    model.save(path)
