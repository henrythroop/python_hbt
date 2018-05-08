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
from   astropy.visualization import wcsaxes
import hbt
from   astropy.wcs import WCS
import matplotlib.pyplot as plt
import matplotlib
import pickle # For load/save
import scipy
import scipy.misc
import os
import hbt

def nh_create_straylight_median(index_group, index_files, do_fft=False, do_sfit=True, power=5):
    
    
    """
    This takes a set of related observations, and creates a median image of all of them,
    in a form useful for straylight removal.
    For now this routine is hard-coded to NH J-ring, but it could be generalized later.
    
    This routine actually does the calculation. 
    
    For a user, should call NH_GET_STRAYLIGHT_MEDIAN, rather than the current function, which will avoid having
    to deal with filenames.
    
    Although the filename is called 'median', this function uses different techniques to calculate the
    best background image, including percentiles, means, etc -- not just a median.
    
    This routine does not use a photoshop mask. That can be used elsewhere.
    
    This routine also does not use an object mask. But, other stray light processing parts do that. 
    
    """

#     o index_group:   What set of observations, grouped by 'Desc' field. Integer.
#     o index_files:   Numpy array list of the files. Cannot be a scalar.
#     o do_fft:        Flag. For the final step, do we use an FFT? [NOT IMPLEMENTED]
#     o do_sfit:       Flag. Do we use an sfit (ie, polynomial fit)?
#     o power:         Exponent for sfit, to be applied at end and subtracted
#   
#     This routine returns the array itself, and a recommended base filename. It does not write it to disk.

    do_debug = True
    
    do_destripe = True
    
    stretch = astropy.visualization.PercentileInterval(90)  # PI(90) scales array to 5th .. 95th %ile

    if do_debug:
        print("nh_create_straylight_median: {}/{}-{}".format(index_group, index_files[0], index_files[-1]))
        
    file_pickle = '/Users/throop/Data/NH_Jring/out/nh_jring_read_params_571.pkl' # Filename to read to get filenames, etc.
    
    lun = open(file_pickle, 'rb')
    t = pickle.load(lun)
    lun.close()

    # Process the group names. Some of this is duplicated logic -- depends on how we want to use it.

    groups = astropy.table.unique(t, keys=(['Desc']))['Desc']
        
    groupmask = (t['Desc'] == groups[index_group])
    t_group = t[groupmask]	
    
    # Create the output arrays
   
    num_files = np.size(index_files)
    
    header = hbt.get_image_header(t_group['Filename'][index_files[0]])
    
    # Get dimensions of first frame in the series
    
    dx_0 = header['NAXIS1']
    dy_0 = header['NAXIS2']
    
    print("For image 0, dx={}".format(dx_0))
    
    frame_arr      = np.zeros((num_files, dx_0, dy_0))
    frame_sfit_arr = np.zeros((num_files, dx_0, dy_0))
    frame_ffit_arr = np.zeros((num_files, dx_0, dy_0))
    
    # Read frames in to make a median
     
    for i,n in enumerate(index_files):
        file = t_group['Filename'][n] # Look up filename
        print("Reading: " + file)
        frame = hbt.read_lorri(file,frac_clip = 1)

        if do_destripe:
            frame = hbt.lorri_destripe(frame)
            
        if (hbt.sizex(frame) != dx_0):
            print("For image {}, dx={}, dy={}".format(i, hbt.sizex(frame), hbt.sizey(frame)))

            print("Error: mixed sizes in this straylight series. Aborting.")
#            raise ValueError('MultipleSizes')
            
            if (np.shape(frame)[0] == 256):
#            
#    Resize the image to 1024x1024, if it is a 4x4 frame. 
#    scipy.misc.imresize should do this, and has various interpolation schemes, but truncates to integer (?!)
#    because it is designed for images. So instead, use scipy.ndimage.zoom, which uses cubic spline.
#        
                frame2 = scipy.ndimage.zoom(frame, 4)
                frame = frame2
                
            if (np.shape(frame)[0] == 1024):
                frame2 = scipy.ndimage.zoom(frame, 0.25)
                frame = frame2
                
        frame_arr[i,:,:] = frame	
        
#        frame_sfit_arr[i,:,:] = frame - hbt.sfit(frame, power)  # Calculate the sfit to each individual frame
#        frame_ffit_arr[i,:,:] = frame - hbt.ffit(frame)
    
    # Now take the median!
    
    frame_med      = np.median(frame_arr, axis=0)
    
    # Now that we have the median for each pixel... take the median of pixels below the median.
    
#    frame_arr_step_2 = frame_arr.copy()
#    for j in range(hbt.sizex(frame_arr)):
#        frame_arr_step_2[j][frame_arr_step_2[j] > frame_med] = np.nan
#    
#    frame_med_step_2 = np.nanmedian(frame_arr_step_2, axis=0)
#
#    frame_arr_step_3 = frame_arr_step_2.copy()
#    for j in range(hbt.sizex(frame_arr)):
#        frame_arr_step_3[j][frame_arr_step_3[j] > frame_med_step_2] = np.nan
#    
#    frame_med_step_3 = np.nanmedian(frame_arr_step_2, axis=0)
#    
#    frame_med_step3  = np.percentile(frame_arr, 10, axis=0)
#    
#    hbt.figsize(25,25)
#    plt.subplot(1,3,1)
#    plt.imshow(stretch(hbt.remove_sfit(frame_med,degree=5)), cmap='plasma')
#    plt.title('Median')
#    
#    plt.subplot(1,3,2)
#    plt.imshow(stretch(hbt.remove_sfit(frame_med_step_2,degree=5)), cmap='plasma')
#    plt.title('Median of pixels < median')
#    
#    plt.subplot(1,3,3)
#    plt.imshow(stretch(hbt.remove_sfit(frame_med_step_3,degree=5)), cmap='plasma')
#    plt.title('Median of pixels < pixels < median')
#    plt.show()
#    hbt.figsize()
    
    #%%%
    
    ######
    # Now just do some experimentation. I am just trying a bunch of things, to try to get the best signal here.
    # In theory a median will work well. The problem is that this just doesn't exclude all the ring.
    # So I tried taking 30th percentile. Works better. 
    # Tried taking 1st percentile (ie, faintest value at each pixel). But this still had a ring, dammit! 
    # I think it must be because a few images are anomalously faint, throwing off the statistics.
    #
    # The method right here uses a very ad-hoc percentile method. There is no reason it should work. 
    # I've just found that it does.
    ######
    
    hbt.figsize(10,10)   
    frame_percentile  = np.percentile(frame_arr, 99, axis=0)
    plt.imshow(stretch(hbt.remove_sfit(frame_percentile,degree=5)), cmap='plasma')

    # Take the mmedians at a bunch of different percentiles
    
    frame_p10  = np.percentile(frame_arr, 10, axis=0)
    frame_p20  = np.percentile(frame_arr, 20, axis=0)
    frame_p30  = np.percentile(frame_arr, 30, axis=0)
    frame_p40  = np.percentile(frame_arr, 40, axis=0)
    frame_p50  = np.percentile(frame_arr, 50, axis=0)
    frame_p60  = np.percentile(frame_arr, 60, axis=0)
    frame_p70  = np.percentile(frame_arr, 70, axis=0)
    frame_p80  = np.percentile(frame_arr, 80, axis=0)
    frame_p90  = np.percentile(frame_arr, 90, axis=0)
    
    frames_pall = np.array([frame_p10, frame_p20, frame_p30, frame_p40, frame_p50, frame_p60, frame_p70, frame_p80])

    # Mean these frames all together
    
    frames_pall_med = np.mean(frames_pall,axis=0)
    
    plt.imshow(stretch(hbt.remove_sfit(frames_pall_med)),cmap='plasma')
    plt.title('frames_pall_med')
    plt.show()
    
    # Save into the ouput array

#    plt.imshow(stretch(hbt.remove_sfit(frame_p90)),cmap='plasma')
    
    frame_med = frames_pall_med

    #%%%

    frame_med_sfit = hbt.remove_sfit(frame_med, degree=power) # Take median using sfit images
    
#    frame_ffit_med = np.median(frame_ffit_arr, axis=0) # Take median using fft  images
    
    # Do a very small removal of hot pixels. Between 0.99 and 0.999 seems to be fine. Make as small 
    # as possible, but something is often necessary
    
#    frame_sfit_med = hbt.remove_brightest(frame_sfit_med, 0.999)
#    frame_ffit_med = hbt.remove_brightest(frame_sfit_med, 0.999)
#    
    file_base = hbt.nh_create_straylight_median_filename(index_group, index_files, do_fft=do_fft, do_sfit=do_sfit, 
                                                     power=power)
                                                     
    if (do_fft): 
        raise ValueError('Sorry, do_ffit not implemented')
        return -1
    
    if (do_sfit):
        return (frame_med_sfit, file_base)
    
    else:
        return (frame_med, file_base)   

# =============================================================================
# End of Function
# =============================================================================
    
if (__name__ == '__main__'):
    
    print("Testing...")
    
    index_group = 8
    index_files = hbt.frange(0,47)
    do_fft      = False
    power       = 5
    do_sfit     = False
    
    (arr,mask) = nh_create_straylight_median(index_group, index_files, do_fft=do_fft, do_sfit=do_sfit, power=power)
 
    plt.imshow(stretch(hbt.remove_sfit(arr, degree=5)),cmap='plasma')
