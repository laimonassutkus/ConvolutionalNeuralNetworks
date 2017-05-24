from tkinter import *
from tkinter import messagebox
from mode import *
import simpleneuralnetwork
import convolutionalneuralnetwork
import largeneuralnetwork
import numpy as np
from graphpainter import paint_number
import threading
import baseneuralnetwork
from statscontainer import Stats, GuessStats

text_box_size = 100
window_width = 300
window_height = 625

root = Tk()
root.geometry('{}x{}'.format(window_width, window_height))
root.resizable(width=False, height=False)

current_mode = Mode.NONE

simple_neural_network = simpleneuralnetwork.SimpleNeuralNetwork()
large_neural_network = largeneuralnetwork.LargeNeuralNetwork()
convolutional_neural_network = convolutionalneuralnetwork.ConvolutionalNeuralNetwork()


def selection_call_back():
    global current_mode
    selection = var.get()
    if selection == 1:
        current_mode = Mode.SNN
        enable_buttons = False if simple_neural_network.is_model_none() else True
    elif selection == 2:
        current_mode = Mode.LNN
        enable_buttons = False if large_neural_network.is_model_none() else True
    elif selection == 3:
        current_mode = Mode.CNN
        enable_buttons = False if convolutional_neural_network.is_model_none() else True
    else:
        current_mode = Mode.ALL
        enable_buttons = False if simple_neural_network.is_model_none() else \
            False if large_neural_network.is_model_none() else \
            False if convolutional_neural_network.is_model_none() else True

    enable_disable_buttons(enable_buttons, True)


def enable_disable_buttons(should_enable, is_mode_selected):
    enabled = 'normal' if should_enable else 'disabled'
    selected = 'normal' if is_mode_selected else 'disabled'

    evaluate_button.config(state=enabled)
    predict_button.config(state=enabled)
    save_button.config(state=enabled)
    train_button.config(state=selected)
    load_button.config(state=selected)


def train_call_back():
    def get_convolutional_model():
        convolutional_neural_network.train_model()
        stats = convolutional_neural_network.get_trained_model_info()
        cnn_info.delete("1.0", END)
        cnn_info.insert(INSERT, "Convolutional neural network:\n" + str(stats))

    def get_neural_model():
        simple_neural_network.train_model()
        stats = simple_neural_network.get_trained_model_info()
        nn_info.delete("1.0", END)
        nn_info.insert(INSERT, "Simple neural network:\n" + str(stats))

    def get_large_model():
        large_neural_network.train_model()
        stats = large_neural_network.get_trained_model_info()
        lnn_info.delete("1.0", END)
        lnn_info.insert(INSERT, "Large neural network:\n" + str(stats))

    if current_mode is Mode.SNN:
        get_neural_model()
        nn_info.config(background='lightgreen')
        enable_disable_buttons(True, True)
    elif current_mode is Mode.LNN:
        get_large_model()
        lnn_info.config(background='lightgreen')
        enable_disable_buttons(True, True)
    elif current_mode is Mode.CNN:
        get_convolutional_model()
        cnn_info.config(background='lightgreen')
        enable_disable_buttons(True, True)
    else:
        messagebox.showinfo('Warning', 'This feature is currently unsupported.')


def show_data_call_back():
    baseneuralnetwork.NeuralNetwork.show_data()


def predict():
    def do_prediction(mnist_image):
        if current_mode is Mode.SNN:
            prediction = simple_neural_network.predict(mnist_image)
            try:
                num_snn = np.where(prediction == 1)[1][0]
                return 'Simple neural network: {}'.format(num_snn)
            except IndexError:
                return "Can't guess."
        elif current_mode is Mode.LNN:
            prediction = large_neural_network.predict(mnist_image)
            try:
                num_snn = np.where(prediction == 1)[1][0]
                return 'Large neural network: {}'.format(num_snn)
            except IndexError:
                return "Can't guess."
        elif current_mode is Mode.CNN:
            prediction = convolutional_neural_network.predict(mnist_image)
            try:
                num_cnn = np.where(prediction == 1)[1][0]
                return 'Convolutional neural network: {}'.format(num_cnn)
            except IndexError:
                return "Can't guess."
        elif current_mode is Mode.ALL:
            prediction1 = simple_neural_network.predict(mnist_image)
            prediction2 = large_neural_network.predict(mnist_image)
            prediction3 = convolutional_neural_network.predict(mnist_image)
            try:
                num_snn = np.where(prediction1 == 1)[1][0]
                num_lnn = np.where(prediction2 == 1)[1][0]
                num_cnn = np.where(prediction3 == 1)[1][0]

                guess_list = [num_cnn, num_lnn, num_snn]
                most_frequent = max(set(guess_list), key=guess_list.count)
                if guess_list.count(most_frequent) is 1:
                    most_frequent = num_cnn

                return 'Simple neural network: {}\n' \
                       'Large neural network: {}\n' \
                       'Convolutional neural network: {}' \
                       '\n--- Overall guess: {} ---'.format(num_snn, num_lnn, num_cnn, most_frequent)

            except IndexError:
                return "Can't guess."

    if current_mode is Mode.SNN:
        paint_number(do_prediction, simple_neural_network.guessStats)
    elif current_mode is Mode.LNN:
        paint_number(do_prediction, large_neural_network.guessStats)
    elif current_mode is Mode.CNN:
        paint_number(do_prediction, convolutional_neural_network.guessStats)
    else:
        paint_number(do_prediction, None)


def predict_call_back():
    nn = threading.Thread(target=predict)
    nn.start()
    # TODO just make an async task - from multiprocessing import Pool
    nn.join()


def save_call_back():
    if current_mode is Mode.SNN:
        simple_neural_network.save('./model.snn')
    elif current_mode is Mode.LNN:
        large_neural_network.save('./model.lnn')
    elif current_mode is Mode.CNN:
        convolutional_neural_network.save('./model.cnn')
    else:
        simple_neural_network.save('./model.snn')
        large_neural_network.save('./model.lnn')
        convolutional_neural_network.save('./model.cnn')


def load_call_back():
    if current_mode is Mode.SNN:
        val = simple_neural_network.load('./model.snn')
        if val is 0:
            nn_info.config(background='lightgreen')
            enable_disable_buttons(True, True)
    elif current_mode is Mode.LNN:
        val = large_neural_network.load('./model.lnn')
        if val is 0:
            lnn_info.config(background='lightgreen')
            enable_disable_buttons(True, True)
    elif current_mode is Mode.CNN:
        val = convolutional_neural_network.load('./model.cnn')
        if val is 0:
            cnn_info.config(background='lightgreen')
            enable_disable_buttons(True, True)
    elif current_mode is Mode.ALL:
        val1 = simple_neural_network.load('./model.snn')
        if val1 is 0:
            nn_info.config(background='lightgreen')
        val2 = large_neural_network.load('./model.lnn')
        if val2 is 0:
            lnn_info.config(background='lightgreen')
        val3 = convolutional_neural_network.load('./model.cnn')
        if val3 is 0:
            cnn_info.config(background='lightgreen')

        if val3 is 0 and val2 is 0 and val1 is 0:
            enable_disable_buttons(True, True)


def evaluate_call_back():
    if current_mode is Mode.SNN:
        simple_neural_network.evaluate()
    if current_mode is Mode.LNN:
        large_neural_network.evaluate()
    elif current_mode is Mode.CNN:
        convolutional_neural_network.evaluate()
    elif current_mode is Mode.ALL:
        simple_neural_network.evaluate()
        convolutional_neural_network.evaluate()


var = IntVar()

chooseSNN = Radiobutton(root, text="Use simple neural network", variable=var, value=1, command=selection_call_back)
chooseSNN.pack()

chooseLNN = Radiobutton(root, text="Use large neural network", variable=var, value=2, command=selection_call_back)
chooseLNN.pack()

chooseCNN = Radiobutton(root, text="Convolutional neural network networks", variable=var, value=3, command=selection_call_back)
chooseCNN.pack()

chooseALL = Radiobutton(root, text="Compare all networks", variable=var, value=4, command=selection_call_back)
chooseALL.pack()

data_samples_frame = Frame(root, height=32, width=window_width)
data_samples_frame.pack_propagate(0)
data_samples_frame.pack()
data_button = Button(data_samples_frame, text="Show data samples", command=show_data_call_back)
data_button.pack(fill=BOTH, expand=1)

train_frame = Frame(root, height=32, width=window_width)
train_frame.pack_propagate(0)
train_frame.pack()
train_button = Button(train_frame, text="Train", command=train_call_back)
train_button.pack(fill=BOTH, expand=1)

save_frame = Frame(root, height=32, width=window_width)
save_frame.pack_propagate(0)
save_frame.pack()
evaluate_button = Button(save_frame, text="Evaluate", command=evaluate_call_back)
evaluate_button.pack(fill=BOTH, expand=1)

predict_frame = Frame(root, height=32, width=window_width)
predict_frame.pack_propagate(0)
predict_frame.pack()
predict_button = Button(predict_frame, text="Predict", command=predict_call_back)
predict_button.pack(fill=BOTH, expand=1)

load_frame = Frame(root, height=32, width=window_width)
load_frame.pack_propagate(0)
load_frame.pack()
load_button = Button(load_frame, text="Load model", command=load_call_back)
load_button.pack(fill=BOTH, expand=1)

save_frame = Frame(root, height=32, width=window_width)
save_frame.pack_propagate(0)
save_frame.pack()
save_button = Button(save_frame, text="Save model", command=save_call_back)
save_button.pack(fill=BOTH, expand=1)

enable_disable_buttons(False, False)

blank = str(Stats(-1, -1, -1, -1, -1))

nn_frame = Frame(root, height=text_box_size, width=window_width)
nn_frame.pack_propagate(0)
nn_frame.pack(padx=5, pady=5)
nn_info = Text(nn_frame)
nn_info.insert(INSERT, "Simple neural network:\n")
nn_info.insert(INSERT, blank)
nn_info.config(background='#%02x%02x%02x' % (240, 240, 240))
nn_info.pack(fill=BOTH, expand=1)

lnn_frame = Frame(root, height=text_box_size, width=window_width)
lnn_frame.pack_propagate(0)
lnn_frame.pack(padx=5, pady=5)
lnn_info = Text(lnn_frame)
lnn_info.insert(INSERT, "Large neural network:\n")
lnn_info.insert(INSERT, blank)
lnn_info.config(background='#%02x%02x%02x' % (240, 240, 240))
lnn_info.pack(fill=BOTH, expand=1)

cnn_frame = Frame(root, height=text_box_size, width=window_width)
cnn_frame.pack_propagate(0)
cnn_frame.pack(padx=5, pady=5)
cnn_info = Text(cnn_frame)
cnn_info.insert(INSERT, "Convolutional neural network:\n")
cnn_info.insert(INSERT, blank)
cnn_info.config(background='#%02x%02x%02x' % (240, 240, 240))
cnn_info.pack()

root.mainloop()
