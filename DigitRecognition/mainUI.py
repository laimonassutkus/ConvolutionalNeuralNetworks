from tkinter import *
from tkinter import messagebox
from mode import *
import neuralnetwork
import convolutionalneuralnetwork
import numpy as np
from graphpainter import paint_number
import threading
import tensorflow as tf

text_box_size = 100
window_width = 300
window_height = 700

root = Tk()
root.geometry('{}x{}'.format(window_width, window_height))
root.resizable(width=False, height=False)

tensorflow_graph = None
current_mode = Mode.NONE

neural_model = None
convolutional_model = None

nn_stats = ["", "", "", "", ""]
cnn_stats = ["", "", "", "", ""]


def selection_call_back():
    global current_mode
    selection = var.get()
    if selection == 1:
        current_mode = Mode.NN
    elif selection == 2:
        current_mode = Mode.CNN
    else:
        current_mode = Mode.BOTH


def train_call_back():
    global neural_model
    global convolutional_model
    global tensorflow_graph
    global nn_stats
    global cnn_stats
    if current_mode == Mode.NN:
        neural_model, nn_stats = neuralnetwork.train_model()
        nn_info.delete("1.0", END)
        nn_info.insert(INSERT, "Simple neural network:\n" + ''.join(str(x) for x in nn_stats))
    elif current_mode == Mode.CNN:
        convolutional_model, cnn_stats = convolutionalneuralnetwork.train_model()
        cnn_info.delete("1.0", END)
        cnn_info.insert(INSERT, "Convolutional neural network:\n" + ''.join(str(x) for x in cnn_stats))
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

chooseNN = Radiobutton(root, text="Use simple neural network", variable=var, value=1, command=selection_call_back)
chooseNN.pack()

chooseCNN = Radiobutton(root, text="Use convolutional neural network", variable=var, value=2, command=selection_call_back)
chooseCNN.pack()

chooseCNN = Radiobutton(root, text="Compare both networks", variable=var, value=3, command=selection_call_back)
chooseCNN.pack()

data_samples_frame = Frame(root, height=32, width=window_width); data_samples_frame.pack_propagate(0); data_samples_frame.pack()
button = Button(data_samples_frame, text="Show data samples", command=show_data_call_back)
button.pack(fill=BOTH, expand=1)

train_frame = Frame(root, height=32, width=window_width); train_frame.pack_propagate(0); train_frame.pack()
button = Button(train_frame, text="Train", command=train_call_back)
button.pack(fill=BOTH, expand=1)

predict_frame = Frame(root, height=32, width=window_width); predict_frame.pack_propagate(0); predict_frame.pack()
button = Button(predict_frame, text="Predict", command=predict_call_back)
button.pack(fill=BOTH, expand=1)

load_frame = Frame(root, height=32, width=window_width); load_frame.pack_propagate(0); load_frame.pack()
button = Button(load_frame, text="Load model", command=load_call_back)
button.pack(fill=BOTH, expand=1)

save_frame = Frame(root, height=32, width=window_width); save_frame.pack_propagate(0); save_frame.pack()
button = Button(save_frame, text="Save model", command=save_call_back)
button.pack(fill=BOTH, expand=1)

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
