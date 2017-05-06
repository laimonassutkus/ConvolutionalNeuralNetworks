from tkinter import *
from tkinter import messagebox
from mode import *
import neuralnetwork
import numpy as np
from graphpainter import paint_number
import threading
import tensorflow as tf


root = Tk()
root.geometry('{}x{}'.format(500, 500))
root.resizable(width=False, height=False)

tensorflow_graph = None
current_mode = Mode.NONE
neural_model = None


def hello_call_back():
    global current_mode
    selection = var.get()
    if selection == 1:
        current_mode = Mode.NN
    elif selection == 2:
        current_mode == Mode.CNN
    else:
        current_mode = Mode.BOTH
    print(selection)


def train_call_back():
    global neural_model
    global tensorflow_graph
    if current_mode == Mode.NN:
        neural_model = neuralnetwork.train_model()
    elif current_mode == Mode.CNN:
        pass
    elif current_mode == Mode.BOTH:
        pass
    else:
        messagebox.showinfo("Error", "Please select the neural network you want to work with!")
    tensorflow_graph = tf.get_default_graph()


def show_data_call_back():
    neuralnetwork.show_data()


def predict():
    def do_prediction(mnist_image):
        mnist_image_vector = mnist_image.reshape(1, 784)  # TODO 784 is hardcoded
        with tensorflow_graph.as_default():
            prediction = neural_model.predict(mnist_image_vector)
            try:
                number = np.where(prediction == 1)[1][0]
                print(number)
            except IndexError:
                pass
    paint_number(do_prediction)


def predict_call_back():
    if neural_model is not None:
        t1 = threading.Thread(target=predict)
        t1.start()
        t1.join()
    else:
        messagebox.showinfo("Error", "Please train the neural network you want to work with or load the model!")


def save_call_back():
    if neural_model is not None:
        neuralnetwork.save('./model.nn', neural_model)
        messagebox.showinfo("Success", "Saved!")
    else:
        messagebox.showinfo("Error", "Please train the neural network first!")


def load_call_back():
    global neural_model
    global tensorflow_graph
    neural_model = neuralnetwork.load('./model.nn')
    tensorflow_graph = tf.get_default_graph()
    messagebox.showinfo("Success", "Loaded!")

var = IntVar()

chooseNN = Radiobutton(root, text="Use simple neural network", variable=var, value=1, command=hello_call_back)
chooseNN.grid(row=0, column=0, columnspan=2, sticky='w')

chooseCNN = Radiobutton(root, text="Use convolutional neural network", variable=var, value=2, command=hello_call_back)
chooseCNN.grid(row=1, column=0, columnspan=2, sticky='w')

chooseCNN = Radiobutton(root, text="Compare both networks", variable=var, value=3, command=hello_call_back)
chooseCNN.grid(row=2, column=0, columnspan=2, sticky='w')

buttonTrain = Button(root, text="Train", command=train_call_back)
buttonTrain.grid(row=3, column=0, columnspan=2, sticky='w')

buttonTrain = Button(root, text="Show data samples", command=show_data_call_back)
buttonTrain.grid(row=4, column=0, columnspan=2, sticky='w')

buttonTrain = Button(root, text="Predict", command=predict_call_back)
buttonTrain.grid(row=5, column=0, columnspan=2, sticky='w')

buttonTrain = Button(root, text="Load model", command=load_call_back)
buttonTrain.grid(row=6, column=0, columnspan=2, sticky='w')

buttonTrain = Button(root, text="Save model", command=save_call_back)
buttonTrain.grid(row=7, column=0, columnspan=2, sticky='w')

root.mainloop()
