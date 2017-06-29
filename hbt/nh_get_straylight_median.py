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

def nh_get_straylight_median(index_group, index_files, do_fft=False, do_sfit=True, power1=5, power2=5):
    
    """
    This is a user-level routine to get a straylight median image.
    If the image exists, it will load it. If it doesn't exist, it will create it.
    This is the one to call under most circumstances.
    """
    
    dir_straylight = '/Users/throop/data/NH_Jring/out/'
    
    file_base = hbt.nh_create_straylight_median_filename(index_group, index_files, 
                                                      do_fft=do_fft, do_sfit=do_sfit, power1=power1, power2=power2)
    
    file_pkl = dir_straylight + file_base + '.pkl'
    
    # If file exists, load it and return
    
    if os.path.exists(file_pkl):
#           print('Reading file: ' + file_pkl)
        lun = open(file_pkl, 'rb')
        arr = pickle.load(lun)
        lun.close() 
        return arr

    # If file doesn't exist, create it, save it, and return
        
    else:
        (arr, file) = hbt.nh_create_straylight_median(index_group, index_files, 
                                                      do_fft=do_fft, do_sfit=do_sfit, power1=power1, power2=power2)

        lun = open(file_pkl, 'wb')
        pickle.dump(arr, lun)
        lun.close()
#           print('Wrote file: ' + file_pkl)
        
        return arr
        
