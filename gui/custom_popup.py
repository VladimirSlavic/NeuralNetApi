import sys
from tkinter import *


class popupWindow(object):
    def __init__(self, master):
        top = self.top = Toplevel(master)
        self.l = Label(top, text="Define architecture: entry X hidden layers X output")
        self.l.pack()
        self.e = Entry(top)
        self.e.pack()
        self.b = Button(top, text='Ok', command=self.cleanup)
        self.b.pack()

    def cleanup(self):
        print("cleanup commencing")
        self.value = self.e.get()
        self.top.destroy()
