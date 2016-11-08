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

def nh_create_straylight_median_filename(index_group, index_files, do_fft=False, do_sfit=True, power1=5, power2=5):
    """
    Just create the base filename for a stray light median file.
    """
    # Usually we will add extensions onto this:
    #  .pkl -- save file
    #  .png -- file to be edited in photoshop
    
    if (do_fft):    #fig, ax = plt.subplots(2, 1, figsize=(20, 15 ))
        file_base = 'straylight_median_g{:.0f}_n{:.0f}..{:.0f}_fft'.format(index_group, np.min(index_files), np.max(index_files))
        return file_base
    
    # NB: The print format string :.0f is useful for taking a float *or* int and formatting as an int.
    #     The 'd' string is for a decimal (int) but chokes on a float.
    
    if (do_sfit):
        if (np.size(index_files) == 0):
            print "Error: index_files = []"
            return ""
            
        file_base = 'straylight_median_g{:.0f}_n{:.0f}..{:.0f}_sfit{:.0f},{:.0f}'.format(index_group, 
                                                             np.min(index_files), np.max(index_files), power1, power2)
        return file_base
        
