# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 14:50:35 2015

@author: throop
"""

import Tkinter
from Tkinter import *

import ttk

import matplotlib.pyplot as plt

import numpy as np
import Tkinter as tk
import matplotlib.figure as mplfig


# Seems to be consensus that I need to set the TkAgg backend
import matplotlib.backends.backend_tkagg as tkagg
# From video. This allows us to not have the standard matplotlib zoom buttons
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg

from matplotlib.figure import Figure

matplotlib.use("TkAgg") # from the pythonprogramming video

pi = np.pi

x0 = -10 * pi
x1 =  10 * pi
numx = 100

win = Tk()
win.fig = plt.Figure(figsize=(10,3))
x = np.arange(x0,x1,numx)
win.fig = plt.plot(x,np.sin(x))


tk.mainloop()


