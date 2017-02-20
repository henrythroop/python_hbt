#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 14:40:49 2017

@author: throop
"""

import pdb
import glob
import math       # We use this to get pi. Documentation says math is 'always available' 
                  # but apparently it still must be imported.
from   subprocess import call
import warnings
import pdb
import os.path
import os
import subprocess

import astropy
from   astropy.io import fits
from   astropy.table import Table
import astropy.table   # I need the unique() function here. Why is in in table and not Table??
import matplotlib.pyplot as plt # pyplot
from   matplotlib.figure import Figure
import numpy as np
import astropy.modeling
from   scipy.optimize import curve_fit
#from   pylab import *  # So I can change plot size.
                       # Pylab defines the 'plot' command
import spiceypy as sp # was cspice
import skimage
from   astropy.wcs import WCS
from   astropy.vo.client import conesearch # Virtual Observatory, ie star catalogs
from   astropy import units as u           # Units library
from   astropy.coordinates import SkyCoord # To define coordinates to use in star search
#from   photutils import datasets
from   astropy.stats import sigma_clipped_stats
from   scipy.stats import mode
from   scipy.stats import linregress
from   photutils import daofind
import wcsaxes
import time
import sys  # For stdout.write, without newline
from scipy.interpolate import griddata

from mpl_toolkits.axes_grid1 import host_subplot # For adding a second axis to a plot
import mpl_toolkits.axisartist as AA             # For adding a second axis to a plot

import imreg_dft as ird
import re # Regexp
import pickle # For load/save
from astropy.io import ascii

# Imports for Tk

from astropy import units as u
from astropy.coordinates import SkyCoord

# HBT imports
import hbt

dir_home = '/Users/throop/git/python_hbt/'
file_df99 = dir_home + 'ccldas_run_data_df99.csv'
file_hbt  = dir_home + 'ccldas_run_data_hbt.csv'


data_df99 = ascii.read(file_df99)
data_hbt  = ascii.read(file_hbt)

# Clip the data so that when we have N=1, the errorbars don't go to infinity

data_hbt['Error'] = np.clip(np.sqrt(data_hbt['Number']), 0.5, 1000)
data_hbt['Error'][0] = 0.7
data_hbt['Error'][1] = 0.7        

# Make a distribution n(r)

q = np.array([2, 3, 4, 5, 5, 6])

numbins = 20
r = hbt.frange(0.001, 1000, numbins, log=True)  # Go from small to big

#==============================================================================
# Make a plot showing my n(r) vs DF99
#==============================================================================

#n = r**(-q)  # This is n(r) in the bin
#ngtr = n.copy()
#for i in range(numbins):
#    ngtr[i] = np.sum(n[i:-1])       # Num > r: sum from here to big

DO_PLOT_ALL_Q = False

if (DO_PLOT_ALL_Q):
    for q_i in q:
        n = r**(-q_i)
        ngtr = n.copy()
        for i in range(numbins):
            ngtr[i] = np.sum(n[i:-1])       # Num > r: sum from here to big
                    
        plt.plot(r, n, label='n(r), q={}'.format(q_i))
        plt.plot(r, ngtr)

plt.rc('font', size=25)
hbt.figsize((12,8))
color_df99 = 'green'
color_hbt = 'purple'

#plt.plot(data_hbt['Diameter'], data_hbt['Number'], label = 'Throop et al 2015', color=color_hbt)
plt.plot(data_df99['Diameter']/2, data_df99['Number'], label = 'Durda & Flynn 1999', color=color_df99)
plt.errorbar(data_hbt['Diameter']/2, data_hbt['Number'], yerr=data_hbt['Error'], label = 'Throop et al 2015', 
             color = color_hbt)

n_q2 = r**(-2)
n_q15 = r**(-1.5)
n_q4 = r**(-4)
plt.plot(r, n_q15*2, color=color_hbt, linestyle='--', alpha = 0.2)
plt.plot(r, n_q4*1000, color=color_df99, linestyle='--', alpha = 0.2)
        
#plt.plot(r, np.cumsum(n))

plt.xscale('log')
plt.yscale('log')
plt.ylabel('N > r')
plt.xlabel('Particle radius r [$\mu$m]')
plt.xlim((0.05, 50))
plt.ylim((5e-5, 200))
plt.legend()
plt.text(5, 2, 'q=4')
#plt.text(0.04, 10, 'q=1.5')
plt.text(20, 0.04, 'q=1.5')


file_out = dir_home + 'ccldas_df99.png'
plt.savefig(file_out)
print("Wrote: " + file_out)
plt.show()




