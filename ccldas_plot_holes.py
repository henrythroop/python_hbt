# -*- coding: utf-8 -*-
"""
Created on Tue Jan  6 21:51:36 2015

@author: throop
"""

# Measure regions for CCLDAS. Better than using ImageJ.
# Idea here is:
#  o Generate SEM images at CSU
#  o Paste them together into a mosaic with Photoshop
#  o Make a new bitmap layer (layer mask, actually) in Photoshop, and mark holes there
#  o Save the layer as a .TIFF file
#  o Read the TIFF file in CCLDAS_MEASURE_REGIONS.PY. Code looks for blobs, and measures their sizes and outputs data to CSV file
#  o Use the current program to import the CSV files and plots the size distributions. 

# This routine works fine on 30K x 30K files. Uses maybe 5-10 GB of RAM but no problems. Might want to reset
# kernel after each run? Not sure.
#
# HBT 5-Jan-2014
 
import numpy as np
import scipy
import pylab
import pymorph
import os
import csv
import glob
from hbt import *

from scipy import ndimage as ndd
from scipy import misc
from pylab import *  # So I can change plot size using rcParams

import matplotlib.pyplot as plt

colors = np.array(['green', 'blue', 'orange', 'brown', 'red', 'pink', 'grey'])

# filestr = '~/Data/CCLDAS/Run3 Nov14/Throop 2014-1-B7/2014-1-B7 Mosaic Good RGB Flat Annotated Holes *.csv'
filestr = '~/Data/CCLDAS/Run3 Nov14/Throop 2014-1-B8/2014-1-B8 Mosaic RGB Annotated Holes *.csv'

files = glob.glob(os.path.expanduser(filestr))

diams_pix = [] # Make a list here. Then we stuff the data from each csv file into this list
diams_cgs = [] # list

num_files = len(files)
files_short = np.zeros(num_files,dtype='S100')

longest = files[0]
for file2 in files:
    longest = longest_common_substring(longest, file2)

file_base = longest.split("/")[-1:][0]  # Extract the common basename of the file, with the pathname removed.

files_uniq = []
for file in files:
    files_uniq.append((file[len(longest):])[:-4])  # remove the common string. Also remove the .csv suffix.
        
# Read in the CSV file and put store all the data into arrays

for file in files:
    print file
    
    f = genfromtxt(file, delimiter=',', skiprows=1, dtype = 'double', invalid_raise=False, usecols=range(4))
    
    if (f.size > 0):
        diam_pix  = f[:,2]
        diam_cgs   = f[:,3]
     
        diams_cgs.append(diam_cgs)
        diams_pix.append(diam_pix)

    else:
        diams_cgs.append(np.array([0]))
        diams_pix.append(np.array([0]))

# Now turn the data into histograms

# Set the parameters for the histogram and size dist
   
um = 1e-4 # um in cgs units
   
nbins = 20
d_min = 0.01 * um
d_max = 20 * um

# Set up the bins themselves. The histograms will go into n(d).

d = np.exp(np.linspace(log(d_min), log(d_max), num=nbins))
n = []

for i in range(num_files):
    ni = np.zeros(nbins) # initialize the distribution for this run

    for j in range(nbins-1):
        ni[j] = np.sum(diams_cgs[i] > d[j]) # Make a cumulative histogram: N > diameter

    n.append(ni)
    
#    
# dist = nd.measurements.histogram(diam_cgs, 0, np.max(diam_cgs[1:]), nbins)

f = 24 # Fontsize, points

rcParams['figure.figsize'] = 15, 10 

# Plot the first line
i = 0
lw = 3

p=plt.plot(d/um,n[i], linewidth=lw, color = colors[i])
for i in range(len(files)):
    plt.plot(d/um, n[i], linewidth=lw, color=colors[i], label = files_uniq[i])
    
#plot = p.add_subplot(111)
plt.xscale('log')
plt.xlim((0.1, 100))
plt.yscale('log')
plt.ylim((0.0001, 200))
plt.xlabel('Hole Diameter [um]', fontsize=f)
plt.ylabel('# > d', fontsize=f)
plt.title(file_base, fontsize=f)
#plt.setp(ax.get_xticklabels(), fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=f * 0.7)
plt.legend()

plt.show()

