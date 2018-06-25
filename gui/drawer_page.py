import os
import threading
from tkinter import *
from tkinter import messagebox

import numpy as np
from PIL import Image, ImageTk
from sklearn.utils import shuffle

from gui.custom_popup import popupWindow
from gui.custom_progress import Progress
from loss_capabilities.quadratic_loss import QuadraticLoss
from loss_capabilities.softmax_loss import SoftMax
from neuralnet.controller import Controller
from neuralnet.fc_net import NeuralNet
from utility.utils import Utiliy


class Drawer:
    def __init__(self, root, M=20):
        self.is_trained = False
        self.root = root
        self.solver = None
        self.points_list = []
        self.M = M
        self.in_process_of_training = False
        self.training_thread = None
        im = Image.open(os.path.join('/home/elrond/PycharmProjects/NENR_LABOS_5/resources', 'eraser.png'))
        im = im.resize((24, 24), Image.ANTIALIAS)
        self.input_dim, self.hidden_layer_dimens, self.output_dimension = 40, [50, 50], 5
        self.erase_image = ImageTk.PhotoImage(im)
        self.test_input = None
        self.activation = 'relu'
        self.loss = SoftMax(one_hot=True)
        self.init_gui()

    def init_gui(self):
        self.root.title("Neuronska mreza: labos 5")
        self.root.resizable(0, 0)

        self.create_menus()
        self.create_toolbar()
        self.create_status_bar()

        self.c = Canvas(self.root, bg="white", width=500, height=500)
        self.c.configure(cursor="crosshair")
        self.c.pack()

        self.c.bind("<Button-1>", self.first_point)
        self.c.bind("<Button-3>", self.clear)
        self.c.bind("<B1-Motion>", self.drag)
        self.c.bind("<ButtonRelease-1>", self.mouse_release)

        self.progress = Progress(self.root)
        self.progress.pb_clear()

        self.save_button = Button(self.root, text='save', relief=RAISED,
                                  command=self.save_info)  # dodat jos command='funkcija'
        self.input_one_hot = Entry(self.root)
        self.label_output_text = StringVar()
        self.output_label = Label(self.root, text='')
        self.predict_button = Button(self.root, text='predict', relief=RAISED,
                                     command=self.make_prediction)  # dodat jos command='funkcija'
        self.train_button = Button(self.root, text="train nnet", command=self.train_net)

        self.save_button.pack(side=LEFT, anchor=W)
        self.input_one_hot.pack(side=LEFT, anchor=W)
        self.output_label.pack(side=RIGHT, anchor=E)
        self.predict_button.pack(side=RIGHT, anchor=E)
        self.input_one_hot.insert(0, '10000')

    def periodiccall(self):
        if self.training_thread.is_alive():
            self.root.after(100, self.periodiccall)
        else:
            self.progress.pb_complete()
            self.is_trained = True
            self.training_thread = None
            self.in_process_of_training = False
            self.final_prediction(self.test_input)

    def create_toolbar(self):
        toolbar = Frame(self.root, bg='grey', relief=RAISED)
        remove_button = Button(toolbar, image=self.erase_image, command=self.clear_toolbar)
        remove_button.pack(side=LEFT, anchor=E, padx=1, pady=1)
        toolbar.pack(side=TOP, fill=X)

    def clear_toolbar(self):
        self.clear(None)

    def create_menus(self):
        menu = Menu(self.root)
        self.root.config(menu=menu)

        file_menu = Menu(menu, tearoff=0)
        menu.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label='Save model', command=self.save_model)
        file_menu.add_command(label='Load model', command=self.load_model)
        file_menu.add_separator()
        file_menu.add_command(label='Define model', command=self.popup)

        datasets_menu = Menu(menu, tearoff=0)
        menu.add_cascade(label='datasets', menu=datasets_menu)
        datasets_menu.add_command(label='iris', command=self.get_iris_data)

        activations_menu = Menu(menu, tearoff=0)
        menu.add_cascade(label='activations', menu=activations_menu)
        activations_menu.add_command(label='sigmoid', command=self.set_sigmoid)
        activations_menu.add_command(label='relu', command=self.set_relu)

        loss_menu = Menu(menu, tearoff=0)
        menu.add_cascade(label='loss types', menu=loss_menu)
        loss_menu.add_command(label='quadratic', command=self.set_quadratic)
        loss_menu.add_command(label='softmax', command=self.set_softmax)

    def popup(self):
        self.w = popupWindow(self.root)
        self.root.wait_window(self.w.top)
        architecture = self.w.value.split('x')
        self.input_dim = int(architecture[0])
        self.hidden_layer_dimens = [int(architecture[index]) for index in range(1, len(architecture) - 1)]
        self.output_dimension = int(architecture[-1])
        print(self.input_dim, self.hidden_layer_dimens, self.output_dimension)

    def set_quadratic(self):
        self.loss = QuadraticLoss(one_hot=True)

    def set_softmax(self):
        self.loss = SoftMax(one_hot=True)

    def set_sigmoid(self):
        self.activation = 'sigmoid'

    def set_relu(self):
        self.activation = 'relu'

    def create_status_bar(self):
        self.status_bar = Label(self.root, text='Nenr labos: neural network', bd=1, relief=SUNKEN, anchor=W)
        self.status_bar.pack(side=BOTTOM, fill=X)

    def save_model(self):
        print('saving model....')

    def load_model(self):
        print('loading model....')

    def get_iris_data(self):
        print('getting iris dataset')

    def reformat_input(self):
        tc_x, tc_y = self.calculate_mean()
        points_normalized = self.recalibrate_points(tc_x, tc_y)
        D = Utiliy.aproximate_distance(points_normalized)
        repr_points = Utiliy.get_representative_points(D, points_normalized, self.M)
        return repr_points

    def save_info(self):
        repr_points = self.reformat_input()
        one_hot_user = self.input_one_hot.get()
        self.input_one_hot.delete(0, END)

        Utiliy.write_to_file(repr_points, one_hot_user)

    def first_point(self, event):
        self.points_list.append((event.x, event.y))  # first added point

    def drag(self, event):
        self.c.create_oval(event.x, event.y, event.x + 1, event.y + 1, fill="black")
        prev_point = self.points_list[len(self.points_list) - 1]
        self.c.create_line(prev_point[0], prev_point[1], event.x, event.y)
        self.points_list.append((event.x, event.y))  # list tuple-a
        return self.points_list

    def calculate_mean(self):
        Tcx = 0.0
        Tcy = 0.0
        # napravi sa list comprehensions
        for elem in self.points_list:
            Tcx += elem[0]
            Tcy += elem[1]
        Tcx /= len(self.points_list)
        Tcy /= len(self.points_list)
        return Tcx, Tcy

    def recalibrate_points(self, tc_x, tc_y):
        points_list_updated = [(elem[0] - tc_x, elem[1] - tc_y) for elem in self.points_list]

        max_y = max(points_list_updated, key=lambda item: abs(item[1]))[1]
        max_x = max(points_list_updated, key=lambda item: abs(item[0]))[0]
        max_ = max(abs(max_x), abs(max_y))

        points_list_updated = [(elem[0] / max_, elem[1] / max_) for elem in points_list_updated]
        return points_list_updated

    def mouse_release(self, event):
        print("mouse released")

    def clear(self, event):
        self.c.delete('all')
        self.points_list.clear()

    def make_prediction(self):

        tc_x, tc_y = self.calculate_mean()
        points_normalized = self.recalibrate_points(tc_x, tc_y)
        D = Utiliy.aproximate_distance(points_normalized)
        draw_pic = Utiliy.get_representative_points(D, points_normalized, self.M)

        converted_result = []
        for elem in draw_pic:
            converted_result.append(elem[0])
            converted_result.append(elem[1])

        self.test_input = converted_result

        if self.in_process_of_training:
            messagebox.showinfo('Neural net occupied', 'Your network is in training, please wait.....')
            return

        if not self.is_trained and self.training_thread is None:
            self.in_process_of_training = True
            self.progress.pb_start()
            self.training_thread = threading.Thread(target=self.train_net)
            self.training_thread.daemon = True
            self.training_thread.start()
            self.periodiccall()
        else:
            self.final_prediction(self.test_input)

    def final_prediction(self, test_input):
        converted_result = np.asarray(test_input, dtype=np.float128)
        converted_result = converted_result.reshape(-1, converted_result.shape[0])
        output = self.solver.predict(np.asarray(converted_result, dtype=np.float128))
        output_text = '00000'
        idx = np.argmax(output, axis=1)[0]
        result = output_text[:idx] + '1' + output_text[(idx + 1):]
        print(result, idx)
        self.output_label['text'] = result
        self.test_input = None

    def train_net(self):
        print("===========================TRAINING NEURAL NET====================================")

        X, y = Utiliy.retrieve_greek_alphabet_data()
        X_train, y_train = shuffle(X, y, random_state=0)

        small_data = {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': None,
            'y_val': None,
        }

        learning_rate = 0.01
        # loss_solver = QuadraticLoss(one_hot=True)  # QuadraticLoss(one_hot=False)

        # implementirat dalje da se u GUI-u zadaju input i output dimensions i radi provjera sa pravim podacima itd.
        model = NeuralNet(hidden_dims=self.hidden_layer_dimens, input_dims=self.input_dim,
                          num_classes=self.output_dimension,
                          loss_type=self.loss, function=self.activation, dtype=np.float128)

        self.solver = Controller(model, small_data,
                                 print_every=1000, num_epochs=20000, batch_size=50,
                                 update_rule='sgd',
                                 optim_config={
                                     'learning_rate': learning_rate,
                                 })
        self.solver.train()
