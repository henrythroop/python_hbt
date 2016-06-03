# -*- coding: utf-8 -*-
"""
Created on Tue May 31 21:13:41 2016

@author: throop
"""

##########
# Read an NH FITS file from disk
##########

import astropy
from astropy.io import fits
import hbt # Seems kind of weird to have to import the module to which this function belongs...
        
def get_image_nh(file, frac_clip=0.9, polyfit=True, bg_method='Polynomial', bg_argument=4):
    " Reads an NH FITS file from disk. Does simple image processing on it, removes background,"
    " and roughly scales is for display."
    " Might still need a log scaling applied for J-ring images."

#    print "get_image_nh, bg_method =" + bg_method
    
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

# Clip any pixels brighter than a specified %ile (e.g., 0.9 = clip to 90th percentile)
               
    arr_clipped = hbt.remove_brightest(arr, frac_clip)
    
# 'None' : If requested, return the raw unscaled image, with no background subtraction

    if (bg_method == 'None'):
        return arr_clipped
               
# 'Polynomial' : Fit the data using a polynomial

    if (bg_method == 'Polynomial'):
        power = int(bg_argument)
        polyfit = hbt.sfit(arr_clipped, power)        
        return arr_clipped - polyfit

# 'Previous' : Subtract off the previous frame (which is passed in as an argument)

    if (bg_method == 'Previous') or (bg_method == 'Next') or (bg_method == 'Median'):
        arr2 = bg_argument
        arr2_clipped = hbt.remove_brightest(arr2, frac_clip)
        return arr_clipped - arr2_clipped
        