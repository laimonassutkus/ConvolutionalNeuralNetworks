from abc import ABC, abstractmethod, abstractproperty
from keras.datasets import mnist
import matplotlib.pyplot as plt
import random
from tkinter import messagebox
from keras.utils import np_utils
from keras.models import load_model
from tkinter import messagebox
import tensorflow as tf


class NeuralNetwork(ABC):
    _trained_model = None
    _trained_model_info = None
    _tf_graph = tf.get_default_graph()

    def is_model_none(self):
        return self._trained_model is None

    def get_trained_model(self):
        if self._trained_model is None:
            messagebox.showinfo('Info', 'Model is not trained. Training...')
            self.train_model()
            return self._trained_model
        else:
            return self._trained_model

    def get_trained_model_info(self):
        if self._trained_model_info is None:
            messagebox.showinfo('Info', 'No info found. Training...')
            self.train_model()
            return self._trained_model_info
        else:
            return self._trained_model_info

    def predict(self, image):
        with self._tf_graph.as_default():
            model = self.get_trained_model()
            return model.predict(self.reshape(image))

    @staticmethod
    def show_data():
        # import (download) data from MNIST
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

        plt.figure()

        plt.subplot(131)
        plt.imshow(X_train[random.randrange(len(X_train))], cmap=plt.get_cmap('gray'))
        plt.subplot(132)
        plt.imshow(X_train[random.randrange(len(X_train))], cmap=plt.get_cmap('gray'))
        plt.subplot(133)
        plt.imshow(X_train[random.randrange(len(X_train))], cmap=plt.get_cmap('gray'))

        # show the plot
        plt.suptitle('Training data examples')
        plt.show()

    def load(self, path):
        self._trained_model = load_model(path)
        messagebox.showinfo("Success", "Successfully loaded model!")

    def save(self, path):
        model = self.get_trained_model()
        model.save(path)
        messagebox.showinfo("Success", "Saved neural model!")

    @abstractmethod
    def reshape(self, image):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def train_model(self):
        pass
