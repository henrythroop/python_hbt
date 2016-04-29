# Simple program to plot histogram of LCOGT comet sub-image
# For Justin Harrison
# HBT 25-Sep-2015

import pdb
import glob
import os.path
from   subprocess import call
import astropy
from   astropy.io import fits
import matplotlib.pyplot as plt # pyplot
import numpy as np
from   pylab import *  # So I can change plot size: 

dir_data = "/Users/throop/Dropbox/data/LCOGT Comet Oct14/"

# Set the file

file = 'coj1m003-kb71-20141017-0105-e90.fits'

# Read the file

hdulist = fits.open(dir_data + '/' + file)

# Read the full image and plot it

plt.set_cmap('gray')
a = hdulist[0].data
hdulist.close()
image = plt.imshow(a)

plt.show()

# Extract comet at center

center = a[970:1070, 970:1070]
image = plt.imshow(log(center))
plt.show()

# Take a histogram

(num,bins) = np.histogram(center, bins=100)

# Set font size for plots

fs = 15 	# Font size

# Make a plot of the histogram itself, using step() function to plot

step(bins[0:-1], num + 1)
plt.xlabel("Value [DN]", fontsize=fs)
plt.ylabel('# Pixels', fontsize=fs)
plt.yscale('log')
plt.show()

# Make a plot of the cumulative histogram

num_cum = np.cumsum(num)
plt.step(bins[0:-1], num_cum)
plt.xlabel("Value [DN]", fontsize=fs)
plt.ylabel('# Pixels <= DN', fontsize=fs)
plt.yscale('log')
plt.xscale('log')
plt.ylim((1e3, 1.1e4))
plt.xlim((6e2, 2e4))
plt.show()

