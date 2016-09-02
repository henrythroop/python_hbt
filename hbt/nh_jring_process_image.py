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

import os.path


import astropy
from   astropy.io import fits
from   astropy.table import Table
import astropy.table   # I need the unique() function here. Why is in in table and not Table??
import matplotlib.pyplot as plt # pyplot
import numpy as np
import astropy.modeling
#from   pylab import *  # So I can change plot size.
                       # Pylab defines the 'plot' command

import pickle # For load/save

import hbt

##########
# Process Image -- do all of the processing on the current image
##########
         
# Returns an image to display
# Returns an image for science (ie, properly scaled, with extrema removed)
# We want to preserve flux, so we can measure I/F off of the image

                                                                                                                                           
def nh_jring_process_image(image_raw, method, argument, index_group, index_image):
    """Image is an array"""
    
# Load the arrays with all of the filenames

    file_pickle = 'nh_jring_read_params_571.pkl' # Filename to read to get filenames, etc.
    
    lun = open(file_pickle, 'rb')
    t = pickle.load(lun)
    lun.close()

    # Process the group names. Some of this is duplicated logic -- depends on how we want to use it.

    groups = astropy.table.unique(t, keys=(['Desc']))['Desc']
    
#    groupname = 'Jupiter ring - search for embedded moons'
#    groupnum = np.where(groupname == groups)[0][0]
        
    groupmask = (t['Desc'] == groups[index_group])
    t_group = t[groupmask]	

    frac, poly = 0, 0
    
    if (method == 'Previous'):
        file_prev = t_group['Filename'][index_image-1]
#            print "file =      " + filename
        print "file_prev = " + file_prev
        image_bg = hbt.read_lorri(file_prev, frac_clip = 1.0, bg_method = 'None', autozoom=True)
        image_fg = image_raw
        image = image_fg - image_bg

    if (method == 'Next'):
        file_next = t_group['Filename'][index_image+1]
        image_bg = hbt.read_lorri(file_next, frac_clip = 1.0, bg_method = 'None', autozoom=True)
        image_fg = image_raw
        image = image_fg - image_bg
        
    if (method == 'Median'): # XXX not working yet
        file_prev = t_group['Filename'][index_image-1]
        image_bg = hbt.read_lorri(file_prev, frac_clip = 1.0, bg_method = 'None')
        image_fg = image_raw
        image = image_fg - image_bg

    if (method == 'Polynomial'):
        
     power = argument
     image = image_raw - hbt.sfit(image_raw, power) # Look up the exponenent and apply it 
                                            
    if (method == 'Grp Num Frac Pow'):  # Specify to subtract a specified group#/image#, mult factor, and sfit power.
                                        # I thought this would be useful, but it turns out we usually need to subtract
                                        # a median of multiple images -- not just one -- so this is not very useful.
                                        # Plus, the best power is usually 5, and the best frac can be calc'd
                                        # with a linfit.
    
        vars = entry_bg.get().split(' ')
        
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
            
            print "group={}, num={}, frac={}".format(grp, num, frac)
#            print "Group = {}, num{}, Name = {}".format(name_group, num, name)

            name_group = groups[grp]
            groupmask = t['Desc'] == name_group
            group_tmp = t[groupmask]
            filename_bg = group_tmp['Filename'][num]
                                    
            image_fg = image_raw
            image_bg = hbt.read_lorri(filename_bg, frac_clip = 1, bg_method = 'None')
            
            image = image_fg - float(frac) * image_bg                
            image = image - hbt.sfit(image, power)
    
    if (method == 'String'):

# Parse a string like "6/112-6/129", or "129", or "6/114", or "124-129" or "6/123 - 129"
# As of 8-July-2016, this is the one I will generally use for most purposes.
# 'String' does this:
#   o Subtract the bg image made by combining the named frames
#   o Subtract a 5th order polynomial
#   o Filter out the extreme highs and lows
#   o Display it.    

        str = argument
        str2 = str.replace('-', ' ').replace('/', ' ')

        vars = str2.split(' ')

#            print "str = " + repr(str2)
#            print "vars = " + repr(vars)
        
        if (np.size(vars) == 0):
            image = image_raw
            image_processed = image
            return
            
        if (np.size(vars) == 1):
            stray = hbt.nh_get_straylight_median(index_group, [int(vars[0])])  # "122" -- assume current group
            
        if (np.size(vars) == 2):
            stray = hbt.nh_get_straylight_median(index_group, hbt.frange(int(vars[0]), int(vars[1])))  # "122-129" 
                                                                                        # -- assume current group
 
        if (np.size(vars) == 3):
            stray = hbt.nh_get_straylight_median(int(vars[0]), hbt.frange(vars[1], vars[2])) # "5/122 - 129"
            
        if (np.size(vars) == 4):
            stray = hbt.nh_get_straylight_median(int(vars[0]), hbt.frange(vars[1], vars[3])) # "6/122 - 6/129"

# Remove sfit(5) from original

        image_proc = hbt.remove_sfit(image_raw, 5)

# Normalize
     
        (stray_norm, coeffs) = hbt.normalize_images(stray, image_proc) # Normalize the images to each other ()

        print
        print "Normalized: med(stray_norm) = " + repr(np.median(stray_norm)) + ', med(image_proc) = ' + repr(np.median(image_proc))
        print "Normalized: mean(stray_norm) = " + repr(np.mean(stray_norm)) + ', mean(image_proc) = ' + repr(np.mean(image_proc))
        
        image_sub = image_proc - stray_norm
        
        image = image_sub
        
        image_bg = stray_norm
        
    if (method == 'None'):
        image = image_raw

    # Scale image by removing stars, CRs, etc. But be careful not to clamp too much: the whole point is that 
    # we are trying to make the ring bright, so don't reduce its peak!

    image =  hbt.remove_brightest( image, 0.97, symmetric=True)   # Remove positive stars
    
    # Save the image where we can ustse it later for extraction. 

    image_processed = image

# If requested: plot the image, and the background that I remove. 
# Plot to Python console, not the GUI.
    
    DO_DIAGNOSTIC = True
    
    if (DO_DIAGNOSTIC):
        plt.rcParams['figure.figsize'] = 16,16

        im = image_raw

        plt.subplot(3,3,1) # vertical, horizontal, index
        plt.title('image_raw, mean=' + hbt.trunc(np.mean(im),3))
        plt.imshow(im)
        
        im = hbt.remove_extrema(im)

        plt.subplot(3,3,4) # vertical, horizontal, index
        plt.imshow(im)
        plt.title('re(image_raw), mean=' + hbt.trunc(np.mean(im),3))

        im = hbt.remove_sfit(im,5)
        plt.subplot(3,3,7) # vertical, horizontal, index
        plt.imshow(im)
        plt.title('re(image_raw) - s5(imag_raw), mean=' + hbt.trunc(np.mean(im),3))
        
#       
        im = stray_norm
 
        plt.subplot(3,3,2)
        plt.imshow(im)
        plt.title('stray_norm, mean=' + hbt.trunc(np.mean(im),3))

        im = hbt.remove_extrema(im)
        plt.subplot(3,3,5)
        plt.imshow(im)
        plt.title('re(stray_norm), mean=' + hbt.trunc(np.mean(im),3))

        im = hbt.remove_sfit(im,5)
        plt.subplot(3,3,8)
        plt.imshow(im)
        plt.title('re(stray_norm) - s5(stray_norm), mean=' + hbt.trunc(np.mean(im),3))
#
        
        im = image_proc - stray_norm
        plt.subplot(3,3,3)
        plt.imshow(im)
        plt.title('raw - stray, mean=' + hbt.trunc(np.mean(im),3))

        im = hbt.remove_extrema(im)
        plt.subplot(3,3,6)
        plt.imshow(im)
        plt.title('re(raw-stray), mean=' + hbt.trunc(np.mean(im),3))

        im = hbt.remove_sfit(im,5)
        plt.subplot(3,3,9)
        plt.imshow(im)
        plt.title('re(raw-stray) - s5(raw-stray), med ' + hbt.trunc(np.median(im),3))
        
        plt.show()

    return image_processed