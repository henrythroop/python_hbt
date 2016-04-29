# -*- coding: utf-8 -*-
"""
Created on Mon Jan  5 14:45:56 2015

@author: throop
"""

# Measure regions for CCLDAS. Better than using ImageJ.
# Idea here is:
#  o Generate SEM images at CSU
#  o Paste them together into a mosaic with Photoshop
#  o Make a new bitmap layer (layer mask, actually) in Photoshop, and mark holes there
#  o Save the layer as a .TIFF file
#  o Read the TIFF file here. Code looks for blobs, and measures their sizes and outputs data to CSV file
#  o Run another program that imports the CSV files and plots the size distributions. 

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

from scipy import ndimage as nd
from scipy import misc
from pylab import *  # So I can change plot size using rcParams

import matplotlib.pyplot as plt

# First define an incr2cum function

file = '~/Desktop/imagej_mask_testing.tif'
file = '~/Data/CCLDAS/run3 nov14/Throop 2014-1-B7/2014-1-B7 Mosaic Good RGB Flat Annotated Holes Definite Test.tif'
file = '~/Data/CCLDAS/Run3 Nov14/Throop 2014-1-B7/2014-1-B7 Mosaic Good RGB Flat Annotated Holes Foil Definite.tif'
#file = '~/Data/CCLDAS/Run3 Nov14/Throop 2014-1-B7/2014-1-B7 Mosaic Good RGB Flat Annotated Holes Foil Maybe.tif'
#file = '~/Data/CCLDAS/Run3 Nov14/Throop 2014-1-B7/2014-1-B7 Mosaic Good RGB Flat Annotated Holes Wire Maybe.tif'
#file = '~/Data/CCLDAS/Run3 Nov14/Throop 2014-1-B7/2014-1-B7 Mosaic Good RGB Flat Annotated Holes Wire Definite.tif'

#file = '~/Data/CCLDAS/Run3 Nov14/Throop 2014-1-B8/2014-1-B8 Mosaic RGB Annotated Holes Foil Definite.tif'
#file = '~/Data/CCLDAS/Run3 Nov14/Throop 2014-1-B8/2014-1-B8 Mosaic RGB Annotated Holes Foil Maybe.tif'
#file = '~/Data/CCLDAS/Run3 Nov14/Throop 2014-1-B8/2014-1-B8 Mosaic RGB Annotated Holes Wire Maybe.tif'
#file = '~/Data/CCLDAS/Run3 Nov14/Throop 2014-1-B8/2014-1-B8 Mosaic RGB Annotated Holes Wire Definite.tif'


print ("Reading file: " + file)
im = misc.imread(os.path.expanduser(file))

print ("Array size:" + repr(im.shape))

# Convert from pixels to microns
# $$SM_MICRON_BAR 139 . ## For 2014-1-B7, 2K x 2K image, 650x
# $$SM_MICRON_MARKER 10µm

um = 1e-4 # microns

pix2cgs = 10 * um / 139

DO_PLOT = (np.max(im.shape) < 5000) # Only plot if it's a small file!

if (DO_PLOT):
  plt.imshow(im)
  plt.title('Original')
  plt.show()

rcParams['figure.figsize'] = 20, 10 

# Flatten from 3D to 2D

print ("Flattening image to 2D...")
im = im.sum(2)

# Convert to integer

# im = int(im)

if (DO_PLOT):
    plt.imshow(im)
    plt.title('Flattened 3D -> 2D')
    plt.show()

# Set background to 0

print ("Thresholding...")
im = im - np.min(im) 

im = np.abs(im - np.median(im))  # 
    
# Take a binary threshold and display results of that

thresh = 0
im[np.where(im > thresh)] = 1

if (DO_PLOT):
    plt.imshow(im)
    plt.title('Threshold > ' + repr(thresh))
    plt.show()

# Generate a structuring element that will consider features connected even if they touch diagonally:
    
s = [[1,1,1],
     [1,1,1],
     [1,1,1]]
# scipy.ndimage.measurements.label

# Label the regions. This is the key part.
# label() distinguishes between non-zero and zero

print ("Labeling regions...")
    
labeled, num_regions    = nd.measurements.label(im, structure=s) # Returns a tuple.

# Now show the final labeled plot
   
if (DO_PLOT):   
    rcParams['figure.figsize'] = 50, 50 

    plt.imshow(labeled)
    plt.title('Labeled')
    plt.show()

    rcParams['figure.figsize'] = 20, 10 

print ("Number of regions: " + repr(num_regions))     

# Now what I want to do is make a table listing each of the objects found. For each one, list its index, x, y, and total area.
# For some reason this is incredibly slow. I think it is the where() function which just hangs up python on a GB array.

print ("Tabulating results...")

DO_LONG_TABLE  = False      # Long table lists positions as well as size. But takes a very long time to calculate and fails for 30K frames. 
DO_SHORT_TABLE = True       # Short table lists sizes only, but runs quickly.

if (DO_LONG_TABLE):
    formatstr = " {0:6} {1:8} {2:6} {3:6} {4:6} {5:6} {6:6}"
    
    formatstr_header = formatstr.replace("f", "")
        
    print formatstr_header.format("  #", "  #pix", "  dx", "  dy", "  diam", "  x", "  y")
        
    for i in range(num_regions):
    
        npix = labeled[ np.where(labeled == i)].size # Number of elements matching i    
    
        xys = np.where(labeled==i)
        xs = xys[0]
        ys = xys[1]
    
        dx = np.max(xs) - np.min(xs) + 1
        dy = np.max(ys) - np.min(ys) + 1
        
        xc = np.median(xs)
        yc = np.median(ys)
        
        x = np.mean(np.where(labeled==i)[0])
        y = np.mean(np.where(labeled==i)[1])
        
        diam = int(2 * np.sqrt(npix / np.pi))
        
        line = formatstr.format(i, npix, dx, dy, diam, xc, yc)
        print line
    
#    print repr(i) + ": # = " + repr(s) + ", x =" + repr(x) + ", y = " + repr(y)

# Do the analysis, but faster

# Use histogram function to count how many of region 0 [= background], region 1, region 2, etc.

hist = nd.measurements.histogram(labeled, 0, num_regions-1, num_regions)

# Now hist[n] = # of pixels in region n.

diam = 2 * np.sqrt(hist / np.pi)
diam_cgs = diam * pix2cgs
formatstr = " {0:6} {1:8} {2:6} {3:6}"

formatstr_header = formatstr.replace("f", "")    
print formatstr_header.format("  #", "  #pix", "  diam", "  diam_cgs")
for i in range(num_regions):
    line = formatstr.format(i, hist[i], diam[i], diam_cgs[i])
    print line
    
print

print ("Outputting as CSV...")

file_csv = file.replace(".tif", ".csv")

with open(os.path.expanduser(file_csv), 'wb') as csvfile:
    lun = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_NONE)
    lun.writerow(['Number', 'Pixels Total', 'Diameter Pixels', 'Diameter cgs'])
    for i in (range(num_regions)[1:]):           # Ignore the 0th element, which is the background
      lun.writerow([i, hist[i], diam[i], diam_cgs[i]])

print ("Wrote file: " + file_csv)
quit

# Now get the size distribution!

nbins = 20

d_min = 0.01 * um
d_max = 20 * um

d = np.exp(np.linspace(log(d_min), log(d_max), num=nbins))

n = np.zeros(nbins)

for i in range(nbins-1):
    n[i] = np.sum(diam_cgs > d[i]) # Make a cumulative histogram: N > diameter
    
# dist = nd.measurements.histogram(diam_cgs, 0, np.max(diam_cgs[1:]), nbins)

f = 24

rcParams['figure.figsize'] = 15, 10 

p=plt.plot(d/um,n, linewidth=3)
plot = p.add_subplot(111)
plt.xscale('log')
plt.xlim((0.1, 100))
plt.yscale('log')
plt.ylim((0.0001, 200))
plt.xlabel('Hole Diameter [um]', fontsize=f)
plt.ylabel('# > d', fontsize=f)
plt.title('2014-1-B7 Definite', fontsize=f)
#plt.setp(ax.get_xticklabels(), fontsize=20)
plot.tick_params(axis='both', which='major', labelsize=30)
plt.show()

# Now convert this into an exponent q
# First need to convert into a differential histogram too.