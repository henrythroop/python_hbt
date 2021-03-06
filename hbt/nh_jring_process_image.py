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
from skimage.io import imread, imsave   # For PNG reading
import os


import pickle # For load/save

import hbt
from nh_create_straylight_median_filename import nh_create_straylight_median_filename
from nh_jring_mask_from_objectlist import nh_jring_mask_from_objectlist

##########
# Process Image -- do all of the processing on the current image
##########
         
# Returns an image with stray light model removed.
# Stray light removal is two steps:
#   1. Remove high frequency, which is using a model generated by summing other images
#   2. Remove low frequency, by fitting a polynomial to the unmasked dataset (ie, ignoring sats + stars)
#
# The image returned is a science image, with flux (ie, I/F) preserved.
# No clipping or scaling has been done.
# 
# If a mask is specified (e.g., 'mask_7_0' in the passed-in string), then the result is returned as a tuple:
#
#   (image, mask) = nh_jring_process_image([...])     # 'mask' is a boolean array
#
# Otherwise, the result is just a regular array 

def nh_jring_process_image(image_raw, method, vars, index_group=-1, index_image=-1, mask_sfit=None):
    
    """
    Return image with stray light removed. Flux is preserved and no clipping is done.

    Parameters
    -----
    
    image_raw: 
        NumPy array with the data image.

    method:    
        Method of background subtraction, which can be 'Next', 'Prev', 'Polynomial', 'String', 'None', 
        'Grp Num Frac Pow', etc.
        
        In general 'String' is the most flexible, and recommended.
        It can be things like "5/0-10 r3 *2 mask_7_10':
            
        - Make a median of Group 5, Images 0-10
        - Rotate them all by 270 degrees
        - Scale it to the data image
        - Multiply background image by 2
        - Subtract data - background
        - Remove a 5th degree polynomial from the result [always done, regardless]
        - Load the Photoshop-created mask file "mask_7_10" and incorporate via a tuple
        - Return final result

    vars:
        The argument to the 'method'. Can be an exponent, a file number, a string, etc -- arbitrary, as needed. 
             
    index_group:
        Index of current image. Not used except for Next/Prev.

    index_image:
        Index of current group. Not used except for Next/Prev.
        
    mask_sfit:
        An optional mask to be applied when doing the sfit. Ony pixels with True will be used.
        This is to mask out satellites, stars, CRs, etc. so they don't affect the sfit().
             
    """

    stretch_percent = 90    
    stretch = astropy.visualization.PercentileInterval(stretch_percent) # PI(90) scales to 5th..95th %ile.


# Load the arrays with all of the filenames

    dir_out = '/Users/throop/Data/NH_Jring/out/'

    file_pickle = dir_out + 'nh_jring_read_params_571.pkl' # Filename to get filenames, etc.
    
    lun = open(file_pickle, 'rb')
    t = pickle.load(lun)
    lun.close()

    # Initialize variables
    
    DO_MASK = False  # We set this based on whether a mask is passed in or not

    dir_mask_stray = dir_out.replace('out','masks')   # Directory where the mask files are
    
    # Process the group names. Some of this is duplicated logic -- depends on how we want to use it.

    groups = astropy.table.unique(t, keys=(['Desc']))['Desc']


    
    if (index_group != -1):  # Only do this if we actually passed a group in
        groupmask = (t['Desc'] == groups[index_group])
        t_group = t[groupmask]	

        # Look up the filename, in case we need it.
    
        file_image = t_group['Filename'][index_image]
        
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
            image_bg = hbt.sfit(image, power, mask=mask_sfit)
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
            image = image - hbt.sfit(image, power, mask=mask_sfit)
            
        image_processed = image

# =============================================================================
# Do method 'None' (trivial)
# =============================================================================
                  
    if (method == 'None'):
        
        image_processed = image = image_raw

#==============================================================================
# Do method 'String'. Complicated, but most useful.
#
# Parse a string like "6/112-6/129", or "129", or "6/114", or "124-129" 
#        or "6/123 - 129" or "6/123-129 r1 *0.5 p4 mask_7_12"
#                  or "".
#
# Except for the group and image number, the order of thse does not matter.        
#==============================================================================
####        
# As of 8-July-2017, this is the one I will generally use for most purposes.
#        
# 'String' does this:
#   o Subtract the bg image made by combining the named frames, and rotating and scaling as requested (optional)
#   o Apply a mask file (optional)
#   o Subtract a polynomial (optional). ** As of 17-Nov-2017, sfit is applied only to masked pixels, not full image.
#        
####
        
    if (method == 'String'):

        str = vars

# =============================================================================
#          Parse any rotation angle -- written as "r90" -- and remove from the string
# =============================================================================
                
        angle_rotate_deg = 0
        
        match = re.search('(r[0-9]+)', str)
        if match:
            angle_rotate_deg = int(match.group(0).replace('r', ''))   # Extract the rotation angle
            str = str.replace(match.group(0), '')                     # Remove the whole phrase from string 
            
            if (np.abs(angle_rotate_deg)) <= 10:      # Allow value to be passed as (1,2,3) or (90, 180, 270)
                angle_rotate_deg *= 90

        # Determine how much the bg frame should be scaled, to match the data frame. This is just a multiplicative
        # factor that very crudely accomodates for differences in phase angle, exptime, etc.
        
# =============================================================================
#          Parse any stray multiplication factor -- written as "*3" -- and remove from the string
#          Multiplicative factor is used to scale the stray light image to be removed, up and down (e.g., up by 3x)
# =============================================================================
        
        factor_stray_default    = 1    # Define the default multiplicative factor
        
        factor_stray            = factor_stray_default
        
        match = re.search('(\*[0-9.]+)', str) # Match   *3   *0.4   etc  [where '*' is literal, not wildcard]
        if match:
            factor_stray = float(match.group(0).replace('*', ''))  # Extract the multiplicative factor
            str = str.replace(match.group(0), '').strip()          # Remove phrase from the string                       

# =============================================================================
#          Parse any mask file -- written as "mask_7_0" -- and remove from the string
# =============================================================================
# 
# This mask is a fixed pattern, read from a file, for stray light etc.
# It is *not* for stars or satellites, which are calculated separately.
#
# To make these mask files, the steps are...

# Maskfile is using same name structure as for the summed bg straylight images -- e.g., 8/0-48.            
#         
# True = good pixel. False = bad.
        
        file_mask_stray = None
                
        match = re.search('(mask[0-9a-z._\-]+)', str)
        
        print("Str = {}, dir_mask_stray = {}".format(str, dir_mask_stray))
        
        if match: 
            file_mask_stray = dir_mask_stray + match.group(0) + '.png'    # Create the filename
            DO_MASK = True
            
            str = str.replace(match.group(0), '').strip()     # Remove the phrase from the string
    
# =============================================================================
#          Parse any polynomial exponent -- written as 'p5'
#          This is the polynomial removed *after* subtracting Image - Stray
# =============================================================================
        
        poly_after_default = 0              # Define the default polynomial to subtract.
                                            # I could do 5, or I could do 0.
                                            
        poly_after         = poly_after_default 
        
        match = re.search('(p[0-9]+)', str)
        if match:
            poly_after = int(match.group(0).replace('p', ''))  # Extract the polynomal exponent
            str = str.replace(match.group(0), '').strip()      # Remove the phrase from the string
            
# =============================================================================
#          Now parse the rest of the string
# =============================================================================

# The only part that is left is 0, 1, or 2 integers, which specify the stray light file to extract
# They must be in the form "7/12-15", or "7/12" or "12"
            
        str2 = str.replace('-', ' ').replace('/', ' ').replace('None', '') # Get rid of any punctuation

        vars = np.array(str2.split(), dtype=int)  # With no arguments, split() breaks at any set of >0 whitespace chars.

# =============================================================================
# Now load the appropriate stray light image, based on the number of arguments passed
# =============================================================================

        do_sfit_stray = False  # Flag: When constructing the straylight median file, do we subtract polynomial, or not?
                               #
                               # Usually we want False. ie, want to do:
                               #  out = remove_sfit(raw - stray)
                               #      not
                               #  out = remove_sfit(raw) - remove_sfit(stray)
                               
        if (np.size(vars) == 0):             #  "<no arguments>"
            image           = image_raw
            image_processed = image
            image_stray     = 0 * image
                    
        if (np.size(vars) == 1):             #  "12" -- image number
            image_stray = hbt.nh_get_straylight_median(index_group, [int(vars[0])],
                                                      do_sfit=do_sfit_stray)  # "122" -- assume current group
            
        if (np.size(vars) == 2):             #  "7-12" -- image range
            image_stray = hbt.nh_get_straylight_median(index_group, 
                                                      hbt.frange(int(vars[0]), int(vars[1])).astype('int'),
                                                      do_sfit=do_sfit_stray)  # "122-129"
                                                                                        # -- assume current group
 
        if (np.size(vars) == 3):             #  "7/12-20" -- group, plus image range
            image_stray = hbt.nh_get_straylight_median(int(vars[0]), 
                                                      hbt.frange(vars[1], vars[2]).astype('int'), 
                                                      do_sfit=do_sfit_stray) # "5/122 - 129"
            
        if (np.size(vars) == 4):             #  "7/12 - 7/20"  (very wordy -- don't use this)
            image_stray = hbt.nh_get_straylight_median(int(vars[0]), 
                                                      hbt.frange(vars[1], vars[3]).astype('int'), 
                                                      do_sfit=do_sfit_stray) # "6/122 - 6/129"


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

#=============================================================================
# Now that we have parsed the string, do the image processing
#=============================================================================

# Load the Photoshop stray mask file, if it exists. Otherwise, make a blank mask of True.
        
        if file_mask_stray:

            try:
                mask_stray = imread(file_mask_stray) > 128      # Read file. Mask PNG file is 0-255. Convert to boolean.
        
                print("Reading mask file {}".format(file_mask_stray))
                
                if (len(np.shape(mask_stray)) > 2):          # If Photoshop saved multiple planes, then just take first
                    mask_stray = mask_stray[:,:,0]
                    
            except IOError:                                   # If mask file is missing
                print("Stray light mask file {} not found".format(file_mask_stray))
            
        else:
            mask_stray = np.ones(np.shape(image_raw),dtype=bool)

# Load the object mask. This masks out stars and satellites, which should not have sfit applied to them.
       
        file_objects = os.path.basename(file_image).replace('.fit', '_objects.txt')
        mask_objects = nh_jring_mask_from_objectlist(file_objects)
        mask_objects = np.logical_not(mask_objects)   # Make so True = good pixel
        
# Merge the two masks together
        
        mask = np.logical_and(mask_objects, mask_stray)  # Output good if inputs are both good

# Rotate the stray light image, if that has been requested 
# [this probably doesn't work, but that's fine -- I didn't end up using this.]

        image_stray = np.rot90(image_stray, angle_rotate_deg/90)  # np.rot90() takes 1, 2, 3, 4 = 90, 180, 270, 360.
        
# Subract the final background image from the data image
        
        image_processed = image_raw - factor_stray * image_stray    

#        print("Removing bg. factor = {}, angle = {}".format(factor_stray, angle_rotate_deg))
        
# Apply the mask: convert any False pixels to NaN in prep for the sfit
            
        image_masked                = image_processed.copy()
        image_masked[mask == False] = math.nan
        
        frac_good = np.sum(mask) / (np.prod(np.shape(mask)))
        
        print("Applying mask, fraction good = {}".format(frac_good))
                
# Remove a polynomial from the result. This is where the mask comes into play.
# XXX NB: I think the logic here could be cleaned up. sfit() now allows a mask= argument ,
#         but it must not have when I wrote this code.        
        
        sfit_masked = hbt.sfit(image_masked, poly_after)
        
        image_processed = image_processed - sfit_masked

        print("Removing sfit {}".format(poly_after))

        # Plot the masks and sfits, for diagnostics 
        
        do_plot_masks = False
        if do_plot_masks:
            
            plt.subplot(1,3,1)
            plt.imshow(stretch(mask))
            plt.title('mask')
            plt.subplot(1,3,2)
            plt.imshow(stretch(image_masked))
            plt.title('image_masked')
            plt.subplot(1,3,3)
            plt.imshow(sfit_masked)
            plt.title('sfit_masked')
            plt.show()
        
# =============================================================================
# END OF CASE STATEMENT FOR METHODS
# =============================================================================

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

        im = hbt.remove_sfit(im, degree=5)
        plt.subplot(1,3,1) # vertical, horizontal, index
        plt.imshow(stretch(hbt.remove_sfit(im, degree=5)))
        plt.title('remove_sfit(image_raw, degree=5), mean=' + hbt.trunc(np.mean(im),3))
        plt.colorbar()
        
#       Column 2: Stray only. This will throw an error if we haven't read in a stray light file -- just ignore it.
        
        plt.subplot(1,3,2)
        try:
            plt.imshow(stretch(hbt.remove_sfit(image_stray, degree=5))) # This won't do much since it is already applied
        except UnboundLocalError:
            print("No stray light to subtract")
        
        plt.title('remove_sfit(stray_norm, degree=5), mean=' + hbt.trunc(np.mean(im),3))

        # Column 3: raw - stray

        plt.subplot(1,3,3)

        try:
            im = hbt.remove_sfit(image_raw - image_stray,degree=5)
            plt.imshow(stretch(im))
        except UnboundLocalError:  
            print("No stray light to subtract")
        
        plt.title('remove_sfit(image_raw - image_stray, degree=5), med ' + hbt.trunc(np.median(im),3))
        
        plt.show()

# Now return the array. If we have a mask, then we return it too, as a tuple
        
    if (DO_MASK): # If we loaded a mask
        return (image_processed, mask)
    else:
        return image_processed

# =============================================================================
# Now do some q&d testing
# =============================================================================

if (__name__ == '__main__'):
    
#    import hbt
    
#    from nh_create_straylight_median_filename import nh_create_straylight_median_filename
    
#    from nh_jring_process_image import nh_jring_process_image
#    from nh_jring_mask_from_objectlist import nh_jring_mask_from_objectlist
    import os.path

    # Set up groups so we can read an image given a group / image number
    
    file_pickle = '/Users/throop/Data/NH_Jring/out/nh_jring_read_params_571.pkl' # Filename to read to get files, etc.

    stretch_percent = 90    
    stretch = astropy.visualization.PercentileInterval(stretch_percent) # PI(90) scales to 5th..95th %ile.

    lun = open(file_pickle, 'rb')
    t = pickle.load(lun)
    lun.close()

    # Process the group names. Some of this is duplicated logic -- depends on how we want to use it.

    groups = astropy.table.unique(t, keys=(['Desc']))['Desc']
    
    index_group = 7
    index_images = [15,16]
    
    method = 'String'
    vars   = 'mask_7_8-15 p5'
#    vars = '64-66 p10 *1.5 mask_7_61-63'
    
    groupmask = (t['Desc'] == groups[index_group])
    t_group = t[groupmask]
    index_images_stray = hbt.frange(8,15)

    # Set up the stray light mask file (created w Photoshop)
    
    file_mask_stray = '/Users/throop/Data/NH_Jring/masks/mask_{}_{}-{}.png'.format(index_group, 
                                                              np.min(index_images_stray),
                                                              np.max(index_images_stray))

    dir_backplanes = '/Users/throop/data/NH_Jring/out/'

    # Now loop over the images
    
    for index_image in index_images:
        file_image = t_group[index_image]['Filename']
        
        # Load the object mask
        
        file_objects = os.path.basename(file_image).replace('.fit', '_objects.txt')
        
        mask_objects = nh_jring_mask_from_objectlist(file_objects)    
    
        do_sfit = False
        
        file_straylight_median = \
          nh_create_straylight_median_filename(index_group, index_image, do_fft=False, do_sfit=do_sfit, power=5)
        
        image_stray = hbt.nh_get_straylight_median(index_group, index_images_stray.astype('int'))
        
        # Read the image
        
        image_raw = hbt.read_lorri(file_image)      
        out       = hbt.nh_jring_process_image(image_raw, method, vars, index_group, index_image)
    
        # Extract the output. There is a second field passed back, if we passed a mask to the input
            
        if len(out) == 2:
            (im, ma) = out
    
        else:
            im = out
    
        # Plot the image!
        
        plt.subplot(1,4,1)
        plt.imshow(stretch(image_raw))
        plt.title('stretch(image_raw)')

        plt.subplot(1,4,2)
        plt.imshow(stretch(im))
        plt.imshow(mask_objects, alpha=0.2, cmap='plasma')
        plt.title('stretch(image_raw), overlay w mask_objects')
        
        plt.subplot(1,4,3)
        plt.imshow(mask_objects)
        plt.title('mask_objects')
        plt.show()

        plt.subplot(1,4,1)
        plt.imshow(stretch(hbt.remove_sfit(image_raw,degree=5)), cmap='plasma')
        plt.title('hbt.remove_sfit(image_raw)')

        plt.subplot(1,4,3)
        plt.imshow(stretch(hbt.remove_sfit(image_raw,degree=5, mask=np.logical_not(mask_objects))), cmap='plasma')
        plt.title('hbt.remove_sfit(mask=mask_objects)')
        
        plt.show()
        
        
###
#    method = 'String'
#    index_group = 5
#    index_image = 2
#    file = '/Users/throop/data/NH_Jring/data/jupiter/level2/lor/all/lor_0034676524_0x630_sci_1_opnav.fit'
#    image_raw = hbt.read_lorri(file)
#    
#    vars = '5/0-5 r3 *0.1'
#    out = nh_jring_process_image(image_raw, method, vars)
#    plt.imshow(stretch(out), origin='lower')
#    plt.show()
#    