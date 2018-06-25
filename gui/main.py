from tkinter import *
from gui.drawer_page import Drawer

root = Tk()
root.title("Neuronska mreza: labos 5")
root.resizable(0, 0)

drawer = Drawer(root=root, M=20)
root.mainloop()
