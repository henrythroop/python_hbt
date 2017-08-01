#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 12:34:07 2016

@author: throop
"""

import pdb
import glob
import math       # We use this to get pi. Documentation says math is 'always available' 
                  # but apparently it still must be imported.


import astropy
from   astropy.io import fits
from   astropy.table import Table
import astropy.table   # I need the unique() function here. Why is in in table and not Table??
import astropy.visualization
import matplotlib.pyplot as plt # pyplot
import numpy as np
import astropy.modeling
import scipy.ndimage
import re

import pickle # For load/save

import hbt

##########
# Process Image -- do all of the processing on the current image
##########
         
# Returns an image with stray light model removed.
# Stray light removal is two steps:
#   1. Remove high frequency, which is using a model generated by summing other images
#   2. Remove low frequency, by fitting a polynomial.
#
# The image returned is a science image, with flux (ie, I/F) preserved.
# No clipping or scaling has been done.

def nh_jring_process_image(image_raw, method, vars, index_group=-1, index_image=-1):
    """Return image with stray light removed. I/F is preserved and no clipping is done.
    
    image_raw: NumPy array with the data image.

    method:    Method of background subtraction:
                  'Next', 'Prev', 'Polynomial', 'String', 'None', 'Grp Num Frac Pow', etc.
               In general 'String' is the most flexible, and recommended.
               It can be things like "5/0-10 r3 *2'
                  - Make a median of Group 5, Images 0-10
                  - Rotate them all by 270 degrees
                  - Scale it to the data image
                  - Multiply by background image by 2
                  - Subtract data - background
                  - Remove a 5th degree polynomail from the result [always done, regardless]
                  - Return final result

    vars:     The argument to the 'method'. Can be an exponent, a file number, a string, etc -- arbitrary, as needed. 
             
    index_group: Index of current image. Not used except for Next/Prev.

    index_image: Index of current group. Not used except for Next/Prev.
             
    """
    
# Load the arrays with all of the filenames

    file_pickle = '/Users/throop/Data/NH_Jring/out/nh_jring_read_params_571.pkl' # Filename to get filenames, etc.
    
    lun = open(file_pickle, 'rb')
    t = pickle.load(lun)
    lun.close()

    # Process the group names. Some of this is duplicated logic -- depends on how we want to use it.

    groups = astropy.table.unique(t, keys=(['Desc']))['Desc']

    if (index_group != -1):  # Only do this if we actually passed a group in
        groupmask = (t['Desc'] == groups[index_group])
        t_group = t[groupmask]	
    
    if (method == 'Previous'):
        file_prev = t_group['Filename'][index_image-1]
#            print "file =      " + filename
        print("file_prev = " + file_prev)
        image_bg = hbt.read_lorri(file_prev, frac_clip = 1.0, bg_method = 'None')
        image_fg = image_raw
        image = image_fg - image_bg
        image_processed = image

    if (method == 'Next'):
        file_next = t_group['Filename'][index_image+1]
        image_bg = hbt.read_lorri(file_next, frac_clip = 1.0, bg_method = 'None', autozoom=True)
        image_fg = image_raw
        image = image_fg - image_bg
        image_processed = image
        
    if (method == 'Median'): # XXX not working yet
        file_prev = t_group['Filename'][index_image-1]
        image_bg = hbt.read_lorri(file_prev, frac_clip = 1.0, bg_method = 'None')
        image_fg = image_raw
        image = image_fg - image_bg
        image_processed = image

    if (method == 'Polynomial'):
        
        power = vars
        image = image_raw - hbt.sfit(image_raw, power) # Look up the exponenent and apply it
        image_processed = image
                                            
    if (method == 'Grp Num Frac Pow'):  # Specify to subtract a specified group#/image#, mult factor, and sfit power.
                                        # I thought this would be useful, but it turns out we usually need to subtract
                                        # a median of multiple images -- not just one -- so this is not very useful.
                                        # Plus, the best power is usually 5, and the best frac can be calc'd
                                        # with a linfit.
            
        if (np.size(vars) == 0): # If no args passed, just plot the image
            power = 0
            frac  = 0
            image = image_raw

        if (np.size(vars) == 1): # One variable: interpret as exponent
            power = float(vars[0])
            frac  = 0
            image = image_raw
            image_bg = hbt.sfit(image, power)
            image = image - image_bg
            
        if (np.size(vars) == 2): # Two variables: interpret as group num and file num
            (grp, num) = vars
            frac  = 1
            power = 0
            
        if (np.size(vars)) == 3: # Three variables: interpret as group num, file num, fraction
            (grp, num, frac) = vars
            power = 0
            
        if (np.size(vars) == 4): # Four variables: Group num, File num, Fraction, Exponent
            (grp, num, frac, power) = vars
             
        if int(np.size(vars)) in [2,3,4]:
           
            grp = int(grp)
            num = int(num)
            frac = float(frac)
            power = int(power)
            
            print("group={}, num={}, frac={}".format(grp, num, frac))
#            print "Group = {}, num{}, Name = {}".format(name_group, num, name)

            name_group = groups[grp]
            groupmask = t['Desc'] == name_group
            group_tmp = t[groupmask]
            filename_bg = group_tmp['Filename'][num]
                                    
            image_fg = image_raw
            image_bg = hbt.read_lorri(filename_bg, frac_clip = 1, bg_method = 'None')
            
            image = image_fg - float(frac) * image_bg                
            image = image - hbt.sfit(image, power)
            
        image_processed = image
    
    if (method == 'String'):

#==============================================================================
# Parse a string like "6/112-6/129", or "129", or "6/114", or "124-129" or "6/123 - 129" or "6/123-129 r1 *0.5"
#                  or "".          
#==============================================================================

####
# As of 8-July-2016, this is the one I will generally use for most purposes.
# 'String' does this:
#   o Subtract the bg image made by combining the named frames. Rotate and scale backgroiund frame as requested.
#   o Subtract a 5th order polynomial
#   o Filter out the extreme highs and lows
#   o Display it.    
####
        str = vars

        # Remove and parse any rotation angle -- written as "6/1-10 r90"
                
        angle_rotate_deg = 0
        
        match = re.search('r([0-9]+)', str)
        if match:
            angle_rotate_deg = int(match.group(0).replace('r', ''))
            str = str.replace(match.group(0), '')
            
            if (np.abs(angle_rotate_deg)) <= 10:      # Allow value to be passed as (1,2,3) or (90, 180, 270)
                angle_rotate_deg *= 90

        # Determine how much the bg frame should be scaled, to match the data frame. This is just a multiplicative
        # factor that very crudely accomodates for differences in phase angle, exptime, etc.
        
        # Parse and remove any stray multiplication factor -- written as "6/1-10 r90 *3"
        
        factor_stray = 1
        
        match = re.search('\*([0-9.]+)', str) # Match   *3   *0.4   etc  [where '*' is literal, not wildcard]
        if match:
            factor_stray = float(match.group(0).replace('*', ''))
            str = str.replace(match.group(0), '')
                           
        str = str.strip() # Remove any trailing spaces left around
        
        # Now parse the rest of the string
        
        str2 = str.replace('-', ' ').replace('/', ' ').replace('None', '')

        vars = np.array(str2.split(), dtype=int)  # With no arguments, split() breaks at any set of >0 whitespace chars.
        
        # Empty string - no arguments
        # This is kind of an extreme case -- e.g., no polynomial subtraction -- so we don't use it much.
        
        if (np.size(vars) == 0):
            image           = image_raw
            image_processed = image
            image_stray     = 0 * image
            
            return image_processed # We want to skip any fitting of the stray, subraction of it, etc. -- so return now.
        
        do_sfit = False
        
        if (np.size(vars) == 1):
            image_stray = hbt.nh_get_straylight_median(index_group, [int(vars[0])],
                                                      do_sfit = do_sfit )  # "122" -- assume current group
            
        if (np.size(vars) == 2):
            image_stray = hbt.nh_get_straylight_median(index_group, 
                                                      hbt.frange(int(vars[0]), int(vars[1])).astype('int'),
                                                      do_sfit=do_sfit)  # "122-129"
                                                                                        # -- assume current group
 
        if (np.size(vars) == 3):
            image_stray = hbt.nh_get_straylight_median(int(vars[0]), 
                                                      hbt.frange(vars[1], vars[2]).astype('int'), 
                                                      do_sfit=do_sfit) # "5/122 - 129"
            
        if (np.size(vars) == 4):
            image_stray = hbt.nh_get_straylight_median(int(vars[0]), 
                                                      hbt.frange(vars[1], vars[3]).astype('int'), 
                                                      do_sfit=do_sfit) # "6/122 - 6/129"


# Adjust the stray image to be same size as original. 
# Sometimes we'll have a 4x4 we want to use as stray model for 1x1 image -- this allows that.
# When we resize it, also adjust the flux (e.g., by factor of 16).

        dx_stray = hbt.sizex(image_stray)
        dx_im    = hbt.sizex(image_raw)
        ratio    = dx_im / dx_stray
        
        if (dx_stray < dx_im):
            image_stray = scipy.ndimage.zoom(image_stray, ratio) / (ratio**2)    # Enlarge the stray image

        if (dx_stray > dx_im):
            image_stray = scipy.ndimage.zoom(image_stray, ratio) / (ratio**2)   # Shrink the stray image
        
# Rotate the image

        image_stray = np.rot90(image_stray, angle_rotate_deg/90)  # np.rot90() takes 1, 2, 3, 4 = 90, 180, 270, 360.
        
# Subtract stray light from original, and then remove an sfit(5) from that

#        print("Raw:   {}, median = {}".format(np.shape(image_raw), np.median(image_raw)))
#        print("Stray: {}, median = {}".format(np.shape(image_stray), np.median(image_stray)))
        
# Scale the stray light image to the data image using linear regression ('normalize'), before removing it.
# Any additional multiplicative factor is on top of this.

        (image_stray_norm, (m,b)) = hbt.normalize_images(image_stray, image_raw)
        image_stray = image_stray_norm
        
#        print("** Normalized stray image with factor m={}, offset b={}".format(m,b))
        
# Subract the final background image from the data image
        
        image_processed = image_raw - factor_stray * image_stray_norm     

# And then remove a polynomial from the result.
        
        power = 5
        
        image_processed = hbt.remove_sfit(image_processed, power)
                        
    if (method == 'None'):
        
        image_processed = image = image_raw

# Remove a small bias offset between odd and even rows ('jailbars')
# This might be better done before the sfit(), but in reality probably doesn't make a difference.

    image_processed = hbt.lorri_destripe(image_processed)
    
# If requested: plot the image, and the background that I remove. 
# Plot to Python console, not the GUI.

# Test stretching here
# We use astropy's stretching here, rather than matplotlib's norm= keyword. The basic idea of both of these 
# is the same, but I know that astropy has a percentile stretch available.

    DO_DIAGNOSTIC = False
    
    if (DO_DIAGNOSTIC):

        stretch = astropy.visualization.PercentileInterval(90)  # PI(90) scales array to 5th .. 95th %ile

        plt.rcParams['figure.figsize'] = 16,6

# Column 1: raw image

        im = image_raw

#        plt.subplot(3,3,1) # vertical, horizontal, index = typewriter-style
#        plt.title('image_raw, mean=' + hbt.trunc(np.mean(im),3))
#        plt.imshow(stretch(im))
#        
#        plt.subplot(3,3,4) # vertical, horizontal
#        plt.imshow(stretch(im))
#        plt.title('image_raw, mean=' + hbt.trunc(np.mean(im),3))

        im = hbt.remove_sfit(im,5)
        plt.subplot(1,3,1) # vertical, horizontal, index
        plt.imshow(stretch(hbt.remove_sfit(im,5)))
        plt.title('image_raw - sfit(imag_raw,5), mean=' + hbt.trunc(np.mean(im),3))
        
#       Column 2: Stray only. This will throw an error if we haven't read in a stray light file -- just ignore it.

#        im = stray
 
#        plt.subplot(3,3,2)
#        plt.imshow(stretch(im))
#        plt.title('stray_norm, mean=' + hbt.trunc(np.mean(im),3))
#
#        plt.subplot(3,3,5)
#        plt.imshow(stretch(im))
#        plt.title('stray_norm, mean=' + hbt.trunc(np.mean(im),3))
        
        plt.subplot(1,3,2)
        try:
            plt.imshow(stretch(hbt.remove_sfit(image_stray,5))) # This won't do much since it is already applied
        except UnboundLocalError:
            print("No stray light to subtract")
        
        plt.title('stray_norm - sfit(stray_norm,5), mean=' + hbt.trunc(np.mean(im),3))

        # Column 3: raw - stray
        
#        im = image_raw - stray_
        
#        plt.subplot(3,3,3)
#        plt.imshow(stretch(im))
#        plt.title('raw - stray, mean=' + hbt.trunc(np.mean(im),3))
#
#        im = hbt.remove_sfit(im,5) - stray_norm
#        
#        plt.subplot(3,3,6)
#        plt.imshow(stretch(im))
#        plt.title('raw-s5(raw) - stray, mean=' + hbt.trunc(np.mean(im),3))

        plt.subplot(1,3,3)

        try:
            im = hbt.remove_sfit(image_raw - image_stray,5)
            plt.imshow(stretch(im))
        except UnboundLocalError:  
            print("No stray light to subtract")
        
        plt.title('sfit(raw-stray,5), med ' + hbt.trunc(np.median(im),3))
        
        plt.show()

    return image_processed

def junk():
    
    method = 'String'
    vars = '7/2-10'
    index_group = 7
    index_image = 2  # This is used only when using Prev / Next. Otherwise it is ignored.

    file = '/Users/throop/data/NH_Jring/data/jupiter/level2/lor/all/lor_0034676524_0x630_sci_1_opnav.fit'

    do_sfit = False
    
    stretch = astropy.visualization.PercentileInterval(90)  # PI(90) scales array to 5th .. 95th %ile

    test = \
      nh_create_straylight_median_filename(index_group, index_image, do_fft=False, do_sfit=do_sfit, power=5)
        
    image_raw = hbt.read_lorri(file)
      
    out = hbt.nh_jring_process_image(image_raw, method, vars, index_group, index_image)

###
    method = 'String'
    index_group = 5
    index_image = 2
    file = '/Users/throop/data/NH_Jring/data/jupiter/level2/lor/all/lor_0034676524_0x630_sci_1_opnav.fit'
    image_raw = hbt.read_lorri(file)
    
    vars = '5/0-5 r3 *0.1'
    out = nh_jring_process_image(image_raw, method, vars)
    plt.imshow(stretch(out))
    plt.show()
    