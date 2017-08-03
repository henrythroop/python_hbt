# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 09:37:07 2016

NH_JRING_CREATE_MEDIANS.PY

This routine creates median files, used for stray light subtraction.

@author: throop
"""

import hbt
import matplotlib.pyplot as plt
import matplotlib
import pickle # For load/save
import scipy
import os

def nh_get_straylight_median(index_group, index_files, do_fft=False, do_sfit=True, power=5):
    
    """
    This is a user-level routine to get a straylight median image (i.e., a specially constructed background image,
    to subtract from a data image).
    If the image exists, it will load it. If it doesn't exist, it will create it.
    This is the one to call under most circumstances.
    The user does not need to worry about the filename of the median image at all. It will just load properly.
    
    do_sfit: Flag. If set, will remove a polynomial from the median. If not, just returns the median itself.
    
    """

    import hbt
    
    dir_straylight = '/Users/throop/data/NH_Jring/out/'
    
# =============================================================================
# Compute the filename
# =============================================================================
    
    file_base = hbt.nh_create_straylight_median_filename(index_group, index_files, 
                                                      do_fft=do_fft, do_sfit=do_sfit, power=power)
    
    file_pkl = dir_straylight + file_base + '.pkl'
    
# =============================================================================
# If median file exists, load it and return
# =============================================================================
    
    if os.path.exists(file_pkl):
        print('Reading file: ' + file_pkl)
        lun = open(file_pkl, 'rb')
        arr = pickle.load(lun)
        lun.close() 
        return arr
        
    else:

# =============================================================================
# If median file doesn't exist, create it and return it
# =============================================================================
        
        (arr, file) = hbt.nh_create_straylight_median(index_group, index_files, 
                                                      do_fft=do_fft, do_sfit=do_sfit, power=power)

        lun = open(file_pkl, 'wb')
        pickle.dump(arr, lun)
        lun.close()
        print('Wrote file: ' + file_pkl)
        
        return arr
   
#==============================================================================
# Test cases
#==============================================================================
     
def test():

    import astropy.visualization
    
    stretch_percent = 90    
    stretch = astropy.visualization.PercentileInterval(stretch_percent) # PI(90) scales to 5th..95th %ile.
    
    arr = hbt.nh_get_straylight_median(6,[120,121,122,123,124],do_sfit=True,  power=5)
    plt.imshow(stretch(arr))
    
    arr = hbt.nh_get_straylight_median(6,[120,121,122,123,124],do_sfit=True, power=10)
    plt.imshow(stretch(arr))

    arr = hbt.nh_get_straylight_median(6,[120,121,122,123,124],do_sfit=False, power=2)
    plt.imshow(arr)    
    
    arr = hbt.nh_get_straylight_median(6,[120,121,122,123,124],do_sfit=True, power=5)
    plt.imshow(arr)
    
    arr_bg = hbt.nh_get_straylight_median(7,[49,50,51],do_sfit=False)
    arr_image = hbt.nh_get_straylight_median(7,[52], do_sfit=False)
    
    plt.imshow(stretch(hbt.remove_sfit(arr_image - arr_bg,1)))
