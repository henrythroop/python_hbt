# -*- coding: utf-8 -*-
"""
Created on Tue May 31 21:13:41 2016

@author: throop
"""

##########
# Read a LORRI FITS file from disk, and optionally do some simple scaling on it 
##########

import astropy
from astropy.io import fits
import numpy as np
import glob
import hbt # Seems kind of weird to have to import the module to which this function belongs...
import scipy.misc
        
# was get_image_nh.py
        
def read_lorri(file, frac_clip=0.9, polyfit=True, bg_method='None', bg_argument=4, autozoom=False):
    """    
    Reads an FITS file from disk. Does simple image processing on it, to scale for display.
    bg_method = None, Polynomial
    """

    dir_images = '/Users/throop/data/NH_Jring/data/jupiter/level2/lor/all'
    
# If there is an empty filename passed, just return an array of random noise, but right size    
    if (file == ''):
        return np.random.random(size=(1024,1024))

# If filename is just an MET (either string or int), then look up the rest of it.

    if hbt.is_number(file):
        file_list = glob.glob(dir_images + '/*{}*fit'.format(int(file)))
        
        if not file_list:
            print 'File not found'
            return 0
        
        file = file_list[0]
        
#    if ('hdulist') in locals():        # If there is already an hdulist, then close it. (Might double-close it, but that is OK.)
#        hdulist.close() 

# If filename has no '/' in it, then prepend the path

    if (dir_images.find('/') == -1):        
        hdulist = fits.open(dir_images + file)
        
    else:
        hdulist = fits.open(file)
        
    image = hdulist['PRIMARY'] # options are 'PRIMARY', 'LORRI Error', 'LORRI Quality'
    arr = image.data

# Close the image
    hdulist.close()

# Zoom it if requested and logical. This expands the 4x4 image to a 1x1.

    if (autozoom) and (np.shape(arr)[0] == 256):
        arr = scipy.ndimage.zoom(arr, 4)

# 'None' : If requested, return the raw unscaled image, with no background subtraction

    if (bg_method.upper() in ['NONE', 'RAW']):
        return arr
        
# Clip any pixels brighter than a specified %ile (e.g., 0.9 = clip to 90th percentile)

    if (frac_clip < 1.):               
        arr_clipped = hbt.remove_brightest(arr, frac_clip)
    else:
        arr_clipped = arr
               
# 'Polynomial' : Fit the data using a polynomial

    if (bg_method.upper() == 'POLYNOMIAL'):
        power = int(bg_argument)
        polyfit = hbt.sfit(arr_clipped, power)        
        return arr_clipped - polyfit