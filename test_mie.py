#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 14:18:22 2016

@author: throop
"""

#==============================================================================
# This is a simple routine to test making phase curves with the pymiecoated library.
# The problem is that that function does not return the phase curve, but only the scattering matrix
# elements, and at one angle at a time.
# 
# The purpose of this routine is to play with the scattering matrices and figure out which combination
# of these yields the phase curve (which I usually call P11).
# 
# Experimentally, I have found that abs(S12[0]**2 + S12[1]**2) looks right.
# S12 is a two-element complex matrix.
#
# HBT 11-Dec-2016
#==============================================================================

import numpy as np
import pymiecoated
import math
import matplotlib.pyplot as plt

angle = np.array(range(100))/100*math.pi

mie = pymiecoated.Mie(x=50,m=complex(1.5,0.00005))

# Create the output arrays

p11  = []
p12  = []
p121 = []

for a in angle:
    S12 = mie.S12(np.cos(a))
    
    p11.append(np.abs(S12[0]**2))
    p12.append(np.abs(S12[1]**2))
#    p121.append(np.abs(S12[0] + S12[1]))
    p121.append(np.abs(S12[0]**2 + S12[1]**2))

plt.plot(angle, p11, label = 'abs(S12[0])')
plt.plot(angle, p12, label = 'abs(S12[1])') # for x=2, looks right
plt.plot(angle, p121, label = 'abs(s12[0] + s12[1]')
plt.yscale('log')    
plt.legend(framealpha=0.5)
