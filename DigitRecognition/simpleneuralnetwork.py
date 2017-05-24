from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from io import StringIO
import sys
import time
from baseneuralnetwork import NeuralNetwork
from tkinter import messagebox
from statscontainer import Stats, GuessStats


class SimpleNeuralNetwork(NeuralNetwork):

    guessStats = GuessStats()

    def reshape(self, image):
        return image.reshape(1, 784)

    def evaluate(self):
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        num_pixels = X_train.shape[1] * X_train.shape[2]
        X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32') / 255
        y_test = np_utils.to_categorical(y_test)

        model = self.get_trained_model()
        scores = model.evaluate(X_test, y_test, verbose=0)
        messagebox.showinfo('Neural Network', 'Baseline Error: %.2f%%' % (100 - scores[1] * 100))

    def train_model(self):
        # import (download) data from MNIST
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

        num_pixels = X_train.shape[1] * X_train.shape[2]
        X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32') / 255
        X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32') / 255

        # convert e.g. 5 to 0000100000
        y_train = np_utils.to_categorical(y_train)
        y_test = np_utils.to_categorical(y_test)

        num_classes = y_test.shape[1]
        # build simple NN model
        def baseline_nn_model():
            model = Sequential()
            # input layer
            model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
            # output layer
            # a softmax activation function is used on the output layer to turn the outputs into probability-like values and
            # allow one class of the 10 to be selected as the model’s output prediction
            model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
            # logarithmic loss is used as the loss function (called categorical_crossentropy in Keras) and the efficient
            # ADAM gradient descent algorithm is used to learn the weights.
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            return model

        model = baseline_nn_model()

        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()
        start = time.time()
        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)
        end = time.time()
        sys.stdout = old_stdout

        stats = mystdout.getvalue().splitlines()[20].split(' ')
        nn_stats = Stats(stats[6], stats[12], stats[3], stats[9], end - start)
        self._trained_model = model
        self._trained_model_info = nn_stats
