from tkinter import *
from tkinter import messagebox
from mode import *
import simpleneuralnetwork
import convolutionalneuralnetwork
import numpy as np
from graphpainter import paint_number
import threading
import tensorflow as tf
import baseneuralnetwork

text_box_size = 100
window_width = 300
window_height = 490

root = Tk()
root.geometry('{}x{}'.format(window_width, window_height))
root.resizable(width=False, height=False)

current_mode = Mode.NONE

simple_neural_network = simpleneuralnetwork.SimpleNeuralNetwork()
convolutional_neural_network = convolutionalneuralnetwork.ConvolutionalNeuralNetwork()

nn_stats = ["", "", "", "", ""]
cnn_stats = ["", "", "", "", ""]


def selection_call_back():
    global current_mode
    selection = var.get()
    if selection == 1:
        current_mode = Mode.NN
        manage_functional_buttons(True)
    elif selection == 2:
        current_mode = Mode.CNN
        manage_functional_buttons(True)
    else:
        current_mode = Mode.BOTH
        manage_functional_buttons(True)
        predict_button.config(state=DISABLED)


def manage_functional_buttons(is_enabled):
    enabled = 'normal' if is_enabled else 'disabled'
    evaluate_button.config(state=enabled)
    predict_button.config(state=enabled)
    save_button.config(state=enabled)
    train_button.config(state=enabled)
    load_button.config(state=enabled)


def train_call_back():
    def get_convolutional_model():
        convolutional_neural_network.train_model()
        stats = convolutional_neural_network.get_trained_model_info()
        cnn_info.delete("1.0", END)
        cnn_info.insert(INSERT, "Convolutional neural network:\n" + ''
                        .join(str(x) for x in stats))

    def get_neural_model():
        simple_neural_network.train_model()
        stats = simple_neural_network.get_trained_model_info()
        nn_info.delete("1.0", END)
        nn_info.insert(INSERT, "Simple neural network:\n" + ''
                        .join(str(x) for x in stats))

    if current_mode is Mode.NN:
        get_neural_model()
        nn_info.config(background='lightgreen')
    elif current_mode is Mode.CNN:
        get_convolutional_model()
        cnn_info.config(background='lightgreen')
    elif current_mode is Mode.BOTH:
        messagebox.showinfo('Warning', 'This feature is currently unsupported.')


def show_data_call_back():
    baseneuralnetwork.NeuralNetwork.show_data()


def predict():

    def do_prediction(mnist_image):
        if current_mode is Mode.NN and not simple_neural_network.is_model_none():
            mnist_image_vector = mnist_image.reshape(1, 784)
            prediction = simple_neural_network.predict(mnist_image_vector)
        elif current_mode is Mode.CNN and not convolutional_neural_network.is_model_none():
            mnist_image_vector = mnist_image.reshape(1, 1, 28, 28)
            prediction = convolutional_neural_network.predict(mnist_image_vector)
        else:
            return "Model not trained."

        try:
            return np.where(prediction == 1)[1][0]
        except IndexError:
            return "Can't guess."

    paint_number(do_prediction)


def predict_call_back():
    if current_mode is Mode.BOTH:
        messagebox.showinfo('Warning', 'This feature is not yet supported.')
        return

    nn = threading.Thread(target=predict)
    nn.start()
    # TODO just make an async task - from multiprocessing import Pool
    nn.join()


def save_call_back():
    if current_mode is Mode.NN and not simple_neural_network.is_model_none():
        simple_neural_network.save('./model.nn')
    elif current_mode is Mode.CNN and not convolutional_neural_network.is_model_none():
        convolutional_neural_network.save('./model.cnn')
    elif current_mode is Mode.BOTH and not convolutional_neural_network.is_model_none() and not simple_neural_network.is_model_none():
        simple_neural_network.save('./model.nn')
        convolutional_neural_network.save('./model.cnn')
    else:
        messagebox.showinfo('Error', 'Network not trained.')


def load_call_back():
    if current_mode is Mode.NN:
        simple_neural_network.load('./model.nn')
        nn_info.config(background='lightgreen')
    elif current_mode is Mode.CNN:
        convolutional_neural_network.load('./model.cnn')
        cnn_info.config(background='lightgreen')
    elif current_mode is Mode.BOTH:
        simple_neural_network.load('./model.nn')
        nn_info.config(background='lightgreen')
        convolutional_neural_network.load('./model.cnn')
        cnn_info.config(background='lightgreen')


def evaluate_call_back():
    if current_mode is Mode.NN:
        simple_neural_network.evaluate()
    elif current_mode is Mode.CNN:
        convolutional_neural_network.evaluate()
    elif current_mode is Mode.BOTH:
        simple_neural_network.evaluate()
        convolutional_neural_network.evaluate()

var = IntVar()

chooseNN = Radiobutton(root, text="Use simple neural network", variable=var, value=1, command=selection_call_back)
chooseNN.pack()

chooseCNN = Radiobutton(root, text="Use convolutional neural network", variable=var, value=2, command=selection_call_back)
chooseCNN.pack()

chooseCNN = Radiobutton(root, text="Compare both networks", variable=var, value=3, command=selection_call_back)
chooseCNN.pack()

data_samples_frame = Frame(root, height=32, width=window_width); data_samples_frame.pack_propagate(0); data_samples_frame.pack()
data_button = Button(data_samples_frame, text="Show data samples", command=show_data_call_back)
data_button.pack(fill=BOTH, expand=1)

train_frame = Frame(root, height=32, width=window_width); train_frame.pack_propagate(0); train_frame.pack()
train_button = Button(train_frame, text="Train", command=train_call_back)
train_button.pack(fill=BOTH, expand=1)

save_frame = Frame(root, height=32, width=window_width); save_frame.pack_propagate(0); save_frame.pack()
evaluate_button = Button(save_frame, text="Evaluate", command=evaluate_call_back)
evaluate_button.pack(fill=BOTH, expand=1)

predict_frame = Frame(root, height=32, width=window_width); predict_frame.pack_propagate(0); predict_frame.pack()
predict_button = Button(predict_frame, text="Predict", command=predict_call_back)
predict_button.pack(fill=BOTH, expand=1)

load_frame = Frame(root, height=32, width=window_width); load_frame.pack_propagate(0); load_frame.pack()
load_button = Button(load_frame, text="Load model", command=load_call_back)
load_button.pack(fill=BOTH, expand=1)

save_frame = Frame(root, height=32, width=window_width); save_frame.pack_propagate(0); save_frame.pack()
save_button = Button(save_frame, text="Save model", command=save_call_back)
save_button.pack(fill=BOTH, expand=1)

manage_functional_buttons(False)

nn_frame = Frame(root, height=text_box_size, width=window_width); nn_frame.pack_propagate(0); nn_frame.pack(padx=5, pady=5)
nn_info = Text(nn_frame)
nn_info.insert(INSERT, "Simple neural network:\n")
nn_info.insert(INSERT, "Accuracy: {}\n".format(nn_stats[0]))
nn_info.insert(INSERT, "Value accuracy: {}\n".format(nn_stats[1]))
nn_info.insert(INSERT, "Loss: {}\n".format(nn_stats[2]))
nn_info.insert(INSERT, "Value loss: {}\n".format(nn_stats[3]))
nn_info.insert(INSERT, "Time to train: {}".format(nn_stats[4]))
nn_info.config(background='#%02x%02x%02x' % (240, 240, 240))
nn_info.pack(fill=BOTH, expand=1)

cnn_frame = Frame(root, height=text_box_size, width=window_width); cnn_frame.pack_propagate(0); cnn_frame.pack(padx=5, pady=5)
cnn_info = Text(cnn_frame)
cnn_info.insert(INSERT, "Convolutional neural network:\n")
cnn_info.insert(INSERT, "Accuracy: {}\n".format(cnn_stats[0]))
cnn_info.insert(INSERT, "Value accuracy: {}\n".format(cnn_stats[1]))
cnn_info.insert(INSERT, "Loss: {}\n".format(cnn_stats[2]))
cnn_info.insert(INSERT, "Value loss: {}\n".format(cnn_stats[3]))
cnn_info.insert(INSERT, "Time to train: {}".format(cnn_stats[4]))
cnn_info.config(background='#%02x%02x%02x' % (240, 240, 240))
cnn_info.pack()

root.mainloop()
