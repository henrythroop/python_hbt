# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 15:52:04 2015

@author: throop
"""
# From http://stackoverflow.com/questions/4073660/python-tkinter-embed-matplotlib-in-gui

import matplotlib, sys
matplotlib.use('TkAgg')
from numpy import arange, sin, pi
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
import numpy as np

import Tkinter
from Tkinter import *
import ttk

global p

def MakePlot(a, c0, c1):
    
    print "MakePlot called"
    
    global p
    
    t = arange(0.0,3.0,0.01) # Time axis
    s = sin(2*pi*(t + c0) * c1)

# Render the plot itself
    p = a.plot(t,s)
    
    return p
   
# 'master' (sometimes called 'root') is now the main address of the window
master = Tk()
master.title("Hello World!")
#-------------------------------------------------------------------------------

# Create a figure, and call it 'f'
# Pass the size of the image here, in something like cm

f = Figure(figsize=(5,4), dpi=100)

# Add a 1x1 image into the Figure (duh)
# http://matplotlib.org/api/figure_api.html for add_subplot args
# The 'a' returned 

a = f.add_subplot(1,1,1, xlabel = 'X', ylabel = 'Y')

# Set up variables for the plot

# t = arange(0.0,3.0,0.01) # Time axis
# s = sin(2*pi*(t + np.random.rand()) * np.random.rand())

# Render the plot itself
# p = a.plot(t,s)
p = MakePlot(a, np.random.rand(), np.random.rand())

# p = a.p # Time, sine
# p.xlabel('X')
#p.ylabel('Y')

# Inside 'master', create the first widget which is a Frame called 'toolbar'

# toolbar = ttk.Frame(master)

# Inside toolbar, create a button
# For some reason, my Frames never render. I am not sure why. I guess
# I won't use those for no! 
# [NB: Did I actually put 'box' into the grid??]

button3 = ttk.Button(master, text='Plot')

# Now make some buttons

button_save = ttk.Button(master, text = 'Save Settings')
button_load = ttk.Button(master, text = 'Load Settings')

# Now make a t
# The figure has been drawn, but here we put it into a Tk container.
# It is apparently not returned as a widget object, since we need to use 
# get_tk_widget() down below for that.

dataPlot = FigureCanvasTkAgg(f, master=master)

# And now render the container
dataPlot.show()

list_dates1 = ['Jan 1', 'Jan 3', 'Jan 17']

list1 = Listbox(master, height=4)
for item in list_dates1:
    list1.insert(END, item)

list_dates2 = ['Jan 2', 'Jan 5', 'Jan 15']

list2 = Listbox(master, height=8)
for item in list_dates2:
    list2.insert(END, item)

list3_functions = ['sin', 'xsin', 'tan', 'xsinx']
list3 = Listbox(master, height=3)
for item in list3_functions:
    list3.insert(END, item)

label0 = Label(master, text = 'Status')
label1 = Label(master, text = 'Date Pluto')
label2 = Label(master, text = 'Date HD')
    
scale = Scale(master, from_=0, to=200, orient=HORIZONTAL)

# Set position for toolbar in the grid

# button.grid(row=0, column=0, sticky="nse")
button3.grid(row=7, column=0, columnspan=2, sticky="")

# Set position for the plot window in the grid
dataPlot.get_tk_widget().grid(row=2, column=0, columnspan=2, sticky="nsew")

scale.grid(row=3, column=0, columnspan=2, sticky = 'ew')
label0.grid(row=1, column=0, columnspan=2)
list1.grid(row=5, column=0, sticky = 'n')
list2.grid(row=5, column=1, sticky = 'n')
label1.grid(row=4, column=0)
label2.grid(row=4, column=1)
list3.grid(row=6, column=0, sticky = 'e')
# box.grid(row=6, column=0, sticky = "ns")


# Set weights for various rows and columns

master.grid_rowconfigure(0, weight=1)
master.grid_rowconfigure(1, weight=1)
master.grid_rowconfigure(2, weight=1)
master.grid_columnconfigure(1, weight=1)
master.grid_rowconfigure(4, weight=1)
master.grid_rowconfigure(3, weight=1)

#l.bind('<Enter>', lambda e: l.configure(text='Moved mouse inside'))

#

# Set up some event handlers

button3.bind('<Button-1>', lambda e: MakePlot(a, np.random.rand(), np.random.rand()))
list2.bind('<Button-1>', lambda e: label0.configure(text='Date HD'))
list1.bind('<Button-1>', lambda e: label0.configure(text='Date Pluto'))

#-------------------------------------------------------------------------------

# Start the loop (which must actually render it)

master.mainloop()