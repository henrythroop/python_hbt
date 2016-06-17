# -*- coding: utf-8 -*-
"""
Created on Tue May 31 21:13:41 2016

@author: throop
"""

##########
# Read a FITS file from disk, and 
##########

import astropy
from astropy.io import fits
import numpy as np
import hbt # Seems kind of weird to have to import the module to which this function belongs...
        
# was get_image_nh.py
        
def get_image_nh(file, frac_clip=0.9, polyfit=True, bg_method='None', bg_argument=4):
    """    
    Reads an FITS file from disk. Does simple image processing on it, to scale for display.
    """
    
# If there is an empty filename passed, just return an array of random noise, but right size    
    if (file == ''):
        return np.random.random(size=(1024,1024))
        
#    if ('hdulist') in locals():        # If there is already an hdulist, then close it. (Might double-close it, but that is OK.)
#        hdulist.close() 
        
    hdulist = fits.open(file)
    image = hdulist['PRIMARY'] # options are 'PRIMARY', 'LORRI Error', 'LORRI Quality'
    arr = image.data

# Close the image
    hdulist.close()

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