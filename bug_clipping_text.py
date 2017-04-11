#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 13:22:54 2017

@author: throop
"""


import numpy as np
import matplotlib.pyplot as plt

import hbt
import astropy.units as u

#%%

hbt.figsize((4,5))
x = hbt.frange(0, 10,100)
y = np.sin(x)

fig, ax = plt.subplots()

xval = 50000*u.km

# Bug: clipping is not applied properly here!
# The xval is converted to a .value when plotted, as expcted.
# But if it has a unit with it, then the clip_on keyword is ignored.
# This causes the plot to be huge, and (I guess) never appears on the screen.

###
#set_clip_on(b)
#Set whether artist uses clipping.
#
#When False artists will be visible out side of the axes which can lead to unexpected results.
###

xval = xval.value    # To trigger bug, comment this line out

ax.plot(x,y)
ax.text(0.5, 0.5, 'adsf')
ax.text(xvalk, -20000, 'XXX', clip_on=True)
ax.set_xlim((-1000,1000))
plt.show()

#%%
