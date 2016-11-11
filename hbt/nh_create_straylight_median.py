# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 09:37:07 2016

NH_JRING_CREATE_MEDIANS.PY

This routine creates median files, used for stray light subtraction.

@author: throop
"""

import math      
import astropy
from   astropy.io import fits
import numpy as np
import spiceypy as sp
import wcsaxes
import hbt
from   astropy.wcs import WCS
import matplotlib.pyplot as plt
import matplotlib
import pickle # For load/save
import scipy
import scipy.misc
import os
import hbt

def nh_create_straylight_median(index_group, index_files, do_fft=False, do_sfit=True, power1=5, power2=5):
    
    """
    This takes a set of related observations, and creates a median image of all of them,
    in a form useful for straylight removal.
    For now this routine is hard-coded to NH J-ring, but it could be generalized later.
    """

#     o group:   What set of observations, grouped by 'Desc' field. Integer.
#     o files:   Int array list of the files 
#     o do_fft:  Flag. For the final step, do we use an FFT or sfit?
#     o do_sfit: Flag.
#     o power1:  Exponent for sfit, to be applied to each frame (step 1)
#     o power2:  Exponent for sfit, to be applied to each frame (step 2)
#   
#     This routine returns the array itself, and a recommended base filename. It does not write it to disk.
 

    file_pickle = '/Users/throop/Data/NH_Jring/out/nh_jring_read_params_571.pkl' # Filename to read to get filenames, etc.
    
    lun = open(file_pickle, 'rb')
    t = pickle.load(lun)
    lun.close()

    # Process the group names. Some of this is duplicated logic -- depends on how we want to use it.

    groups = astropy.table.unique(t, keys=(['Desc']))['Desc']
    
#    groupname = 'Jupiter ring - search for embedded moons'
#    groupnum = np.where(groupname == groups)[0][0]
        
    groupmask = (t['Desc'] == groups[index_group])
    t_group = t[groupmask]	
    
    # Create the output arrays
    # For now I am assuming 1X1... I'll have to rewrite this if I use 4X4's very much.
   
    num_files = np.size(index_files)
    
    frame_arr      = np.zeros((num_files, 1024, 1024))
    frame_sfit_arr = np.zeros((num_files, 1024, 1024))
    frame_ffit_arr = np.zeros((num_files, 1024, 1024))
    
    # Read frames in to make a median
     
    for i,n in enumerate(index_files):
        file = t_group['Filename'][n] # Look up filename
        print("Reading: " + file)
        frame = hbt.read_lorri(file,frac_clip = 1)
        if (np.shape(frame)[0] == 256):
            
    # Resize the image to 1024x1024, if it is a 4x4 frame. 
    # scipy.misc.imresize should do this, and has various interpolation schemes, but truncates to integer (?!)
    # because it is designed for images. So instead, use scipy.ndimage.zoom, which uses cubic spline.
        
            frame2 = scipy.ndimage.zoom(frame, 4)
            frame = frame2
        frame_arr[i,:,:] = frame	
        
        frame_sfit_arr[i,:,:] = frame - hbt.sfit(frame, power1)
        frame_ffit_arr[i,:,:] = frame - hbt.ffit(frame)
    
    frame_sfit_med = np.median(frame_sfit_arr, axis=0) # Take median using sfit images
    frame_ffit_med = np.median(frame_ffit_arr, axis=0) # Take median using fft  images
    
    # Do a very small removal of hot pixels. Between 0.99 and 0.999 seems to be fine. Make as small 
    # as possible, but something is often necessary
    
#    frame_sfit_med = hbt.remove_brightest(frame_sfit_med, 0.999)
#    frame_ffit_med = hbt.remove_brightest(frame_sfit_med, 0.999)
#    
    file_base = hbt.nh_create_straylight_median_filename(index_group, index_files, do_fft=do_fft, do_sfit=do_sfit, 
                                                     power1=power1, power2=power2)
                                                     
    if (do_fft):   
        return (frame_ffit_med, file_base)
    
    if (do_sfit):
        return (frame_sfit_med, file_base)
        
