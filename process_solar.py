#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 22:56:57 2023

@author: throop
"""

# from hbt_short import hbt 

# import hbt_short

import glob
import os.path
import os
import subprocess
import datetime
import matplotlib.pyplot as plt
import math
import numpy as np
import astropy
import astropy.modeling
#import matplotlib.pyplot as plt # Define in each function individually
import matplotlib                # We need this to access 'rc'
# import spiceypy as sp
from   astropy.io import fits
import subprocess
# import hbt
import warnings
import importlib  # So I can do importlib.reload(module)
# from   photutils import DAOStarFinder
from astropy.stats import SigmaClip
from photutils.background import Background2D, MedianBackground
from   astropy import units as u           # Units library
import scipy.ndimage
from astropy.time import Time
from astropy.visualization import PercentileInterval

from   scipy.optimize import curve_fit
from   astropy.convolution import Box1DKernel, Gaussian1DKernel, convolve
from PIL import Image
from scipy import ndimage
from astropy.visualization import simple_norm
from astropy.modeling import models
from astropy.modeling.models import Polynomial1D
from astropy.modeling.fitting import LinearLSQFitter
from astropy.modeling import fitting
from scipy import signal


import scipy.misc
import pytz

from suncalc import get_position, get_times
from datetime import datetime, timezone, timedelta

# https://circuitcellar.com/research-design-hub/projects/field-derotator-for-astrophotography-part-1/
#  OK, this one above has a closed-form eq for the field rotation rate. But as input, it requires the alt + az of the sun. 
#  That is not quite trivial to calculate. 

# https://vixra.org/pdf/2205.0085v1.pdf

# https://learnopencv.com/image-alignment-feature-based-using-opencv-c-python/

# In SPICE what I would do:
#  Set up an observer position in Colombo, on Earth. Get its XYZ.
#  I guess make a new frame called Telescope
#  Set up a rotation matrix from Earth, to Telescope frame. It will have two rotations (plus maybe two to get to Colombo?)
#  Then set up a unit vector
#  Then to PXFORM this unit vector onto the sky plane.
#  Then get VCSP angular difference between this vector, and the sun's polar vector.
#
#  This is conceptually straight forward, but I could spend a week on it.
#
# NB: It is simple to get alt and az of Sun. I can use SPICE for that, as seen from Colombo    

def angle_sun():
    
# I gave up trying to calculate solar position using SPICE! It is really hard.
# Instead, use the 'suncalc' library.     

    d2r = 2*math.pi/360
    r2d = 1/d2r
    lon = 79.853        
    lat = 6.913
    
    get_position(date, lon, lat)
    get_times(date, lon, lat)
    
    date = datetime.now(pytz.timezone('UTC'))  # Input is UTC time, always. Not necessary to pass this, but is OK.
    date = datetime.now()  
    # Date: UTC time
    # Longitude: Degrees. DC = -77 deg. Colombo = +79 deg.
    # Azimuth: -90 = facting east; +90 = facing west. 0 = south??
    # Altitude: degrees above horizon
    # Lon / lat are degrees
    
    result = get_position(date, lon, lat)
    az = result['azimuth']
    alt = result['altitude']
    print('Az = ' + str(az * r2d) + '; Alt = ' + str(alt * r2d))

def test_angle_sun():

    d2r = 2*math.pi/360
    r2d = 1/d2r
    
    date = datetime.now(pytz.timezone('UTC'))  # Input is UTC time, always. Not necessary to pass this, but is OK.
    date = datetime.now()
    
    # Date: UTC time
    # Longitude: Degrees. DC = -77 deg. Colombo = +79 deg.
    # Azimuth: -90 = facting east; +90 = facing west. 0 = south??
    # Altitude: degrees above horizon
    # Lon / lat are degrees
    #
    #   altitude: sun altitude above the horizon in radians, e.g. 0 at the horizon and PI/2 at the zenith 
    #      (straight over your head)
    #   azimuth: sun azimuth in radians (direction along the horizon, measured from south to west), 
    #      e.g. 0 is south and Math.PI * 3/4 is northwest
    
    POS_HOME = 'Colombo'

    if (POS_HOME == 'Colombo'):
        lon         = 79.853        
        lat         = 6.913
        offset_gmt  = 5.5
    
    result = get_position(date, lon, lat)
    az = result['azimuth']
    alt = result['altitude']
    print('Az = ' + str(az * r2d) + '; Alt = ' + str(alt * r2d))
    
    date = datetime.now()
    
    delta = timedelta(seconds=3600*offset_gmt)

    # Verify that sunrise and sunset elevations are correct. Yes they are -- but I need to pass local time, not UTC!
    # This is probably an issue with the timedate() module, moreso than with this code.
    
    get_position(get_times(date, lon, lat)['sunset'] + timedelta(seconds=3600*offset_gmt), lon, lat)
    get_position(get_times(date, lon, lat)['sunrise'] + timedelta(seconds=3600*offset_gmt), lon, lat)
    
    get_times(datetime.now(), lon, lat)

def cross_image(im1, im2):
    # This detects the shift between two images. This is the real meat.
    #   Limb-fit: can't find a good algorhtm for that.
    #   Center-of-mass: tried that on Australia eclipse; it's not good.
    #   Image correlation: This works well! a full-disk images and a small crescent cen be aligned this way, easily.
    
    # Basically the fftconvolve() does a cross-correlate between the two iamges, at all possible shift locations.
    # This is the real bottleneck of the routine. Speed-wise I could rewrite to do this 
    # at lower resolution, or not over the entire image, etc. That would really be superior.
    
    # https://stackoverflow.com/questions/67560829/python-image-processing-how-do-you-align-images-that-have-been-rotated-and-shif
    
    im1_gray = im1.astype('float')
    im2_gray = im2.astype('float')

    im1_gray -= np.mean(im1_gray)
    im2_gray -= np.mean(im2_gray)

    return signal.fftconvolve(im1_gray, im2_gray[::-1,::-1], mode='same')

def getMaskDisk(img): 
    
    # Return a mask which is the solar disk itself
    # Philosophy here: Get the pixels that are at the 99th %ile (ie, definitely on the solar disk),
    # and then back off a bit (to get *all* of the disk).
    # This will faily during totality since zero disk visible, but still works during annularity.

    isDisk = img > np.percentile(img, 99.9)/4 ## Used this at first, for eclipses etc.
    isDisk = img > np.percentile(img, 99.9)/2 ## Using more tight tolerance, for testing

    return(isDisk)

def crop_square(im, dxy):

    # Takes an 2D array (ie, image), and returns a square dxy x dxy extracted from the center
    
    widthX = np.shape(im)[1]
    widthY = np.shape(im)[0]

    centerY = widthY/2
    centerX = widthX/2
    
    sizeYOut = dxy
    sizeXOut = dxy
    
    
    img_out = im[  int(widthY/2 + int(centerY-widthY/2) - sizeYOut/2):
                   int(widthY/2 + int(centerY-widthY/2) + sizeYOut/2),
                   int(widthX/2 + int(centerX-widthX/2) - sizeXOut/2):
                   int(widthX/2 + int(centerX-widthX/2) + sizeXOut/2)]
    
    return(img_out)

    
def process_all():
    
    ## Define the size of the final output image, in pixels
    
    sizeXOut = 2000 # This is arbitrary. For NM, 2000 is a tight crop, but doesn't allow much rotation.
    sizeYOut = 2000
    
    DO_CROP     = True
    DO_ROTATE = False  # Either use SPICE, or https://github.com/cytan299/field_derotator/tree/master/field_derotator_formula
    DO_LIMBFIT  = False
    DO_FLATTEN  = True

    d2r = 2*math.pi/360
    r2d = 1/d2r
#     omegaEarth = 2*math.pi / 86400  # Earth rotation rate, radians/sec.
    omegaEarth = 4.178e-3 * math.pi / 180 ## Rad/sec, fixed

    lon = 79.853        
    lat = 6.913
    
    # Set the path of the files to look at
    
    path_in = '/Volumes/Data/Solar/7Mar24'
    
    path_out = os.path.join(path_in, 'out')
    if not(os.path.exists(path_out)):
        os.mkdir(path_out)
        print('Create: ' + path_out)
        
    files = glob.glob(os.path.join(path_in, '*.fit*'))
    files = np.sort(files)
    
    file = files[0]
    
    plt.set_cmap(plt.get_cmap('Greys_r'))

# Do a first pass through the files, to get start and end times
# We use these to calculate the rotation angle, if we use it.

    t = []
    for file in files:
    
        # Read the original data + header
        
        hdu = fits.open(file)
        header = hdu['PRIMARY'].header
        date_obs = header['DATE-OBS']
        t.append(datetime.strptime(date_obs,"%Y-%m-%dT%H:%M:%S.%f"))
        hdu.close()
        
    t_0 = min(t) # Start time
    t_1 = max(t) # End time
    
    dt  = max(t) - min(t)

    num_t          = int((t_1 - t_0).total_seconds() + 1) # Total duration, from start to end

    print(f'Path: {path_in}') 
    print(f'{len(files)} images found, spanning {num_t} seconds starting at {min(t)}')

  # Now set up an array for rotation. Make one-second time bins.
  
    stepsize_sec   = 1
    
    num_t          = int(((t_1 - t_0).total_seconds() + 1) / stepsize_sec) # Total duration, from start to end
    t_arr          = np.ndarray(int(num_t), dtype=type(t_0))   # Array of datetime objects
    az_arr         = np.zeros(num_t)
    alt_arr        = np.zeros(num_t)
    delta_t_arr    = np.zeros(num_t)
    omegaField_arr = np.zeros(num_t) # Field rotation rate, rad/sec
    angleField_arr = np.zeros(num_t) # Total radians that field has rotated since start
  
    for i in range(0,num_t):

      # Create a new time object for this timestep
      
      # For some reason, need to do a timezone offset. Not sure why.
      
      fudge = 3600*5.5 #  For Colombo
      fudge = 3600*(-6) #  For New Mexico
      
      t_i = t_0 + timedelta(seconds=i*stepsize_sec + fudge)

      # Use the time object to look up the geometry right now
      
      result     = get_position(t_i, lon, lat)
      az_arr[i]  = result['azimuth']
      alt_arr[i] = result['altitude']
      t_arr[i]   = t_i
      omegaField_arr[i] = -omegaEarth * math.cos(az_arr[i]) * math.cos(lat*d2r) / math.cos(alt_arr[i])
      
      # Now finally get our answer! Rotation angle is just sum of all angles up til now
      
      angleField_arr[i] = np.sum(omegaField_arr[0:i]) * stepsize_sec 

    plt.plot(az_arr*r2d, label='Az')
    plt.plot(alt_arr*r2d, label='Alt')
    plt.plot(angleField_arr*r2d, label='Rotate')
    plt.xlabel('Time Step')
    plt.ylabel('Deg')
    plt.legend()
    plt.show()

# Set up image centering, by defining a reference image which is indeed properly centered.

    # For New Mexico:
    # N = 120 = just before start
    # N = 100 = good reference before eclipse
    # N = 350 = at center
    # N = 750 = just before exit
    
    # Load the reference image
    
    index_ref = 0
    hdu = fits.open(files[index_ref])
    img_ref = hdu['PRIMARY'].data
    hdu.close()
    img_ref_mask = getMaskDisk(img_ref)
    
    # Center it
    
    (centerY, centerX) = scipy.ndimage.measurements.center_of_mass(img_ref_mask)
    
    dx_roll = centerX-np.shape(img_ref_mask)[1]/2
    dy_roll = centerY-np.shape(img_ref_mask)[0]/2
    img_ref_mask_cen = np.roll(np.roll(img_ref_mask, -round(dx_roll), axis=1), -round(dy_roll), axis=0)

    # Do the FFT on the centered reference image vs. itself. This is to set up the FFT for next time.
        
    corr_img_null = cross_image(img_ref_mask_cen, img_ref_mask_cen) # Cross-correlate source and target mask
    y0, x0 = np.unravel_index(np.argmax(corr_img_null), corr_img_null.shape) # And get position of peak of that CC.
    
    # Now we have created a centered image reference. We will align all images to this.    
          
# Now do the processing of each image in the sequence
# THIS IS THE MAIN IMAGE LOOP

    
    for i,file in enumerate(files):
        
        string_actions = ''

        # Read the data + header
        
        hdu = fits.open(file)
        img_i = hdu['PRIMARY'].data
        hdu.close()
        
        # Now make a copy of the img_i. We use this as the 'running' copy through all of the steps.
        
        img_0 = img_i.copy() # Save the original, so we can compare
        
        img = img_i.copy()
        img_mask = getMaskDisk(img)
        
        if DO_CROP:       ## Do the centroid + crop, if requested
            
            # Calculate the offset between image and reference. We do this using the binary mask of the image,
            # rather than the images themselves. This routine is magic, and just gives the right answer, every time.
            
            corr_img = cross_image(img_mask, img_ref_mask_cen)
            y, x     = np.unravel_index(np.argmax(corr_img), corr_img.shape)
                
            ver_shift = y0-y
            hor_shift = x0-x
            
            # Plot the recentered mask image, on top of the reference mask
            # plt.imshow(img_ref_mask_cen - img_i_mask_cen/2)
            
            # Now apply this offset to the actual image, not the mask
            
            img_cen = np.roll(np.roll(img, hor_shift, axis=1), ver_shift, axis=0)
            img_cen_crop = crop_square(img_cen, sizeXOut)
                
            img_c = img_cen_crop
            
            # Now create the image for the next step
            
            img = img_c.copy()
            
            string_actions = string_actions + 'c'
            
            # img_pil_c = Image.fromarray(img_i_cen_crop)    
                
        # Do some additional processing, if requested: Rotate, flatten, etc.
    
        if (DO_FLATTEN):
        
            # Remove a background gradient from the image
            # Add a suffix '_f', for 'flattened'
            
            img_mask = getMaskDisk(img)
            
            bkg_estimator = MedianBackground()
            
            bkg = Background2D(img, (50, 50), filter_size=(3, 3), bkg_estimator=bkg_estimator, mask=img_mask)
        
            img_f = (img - bkg.background)

            plt.imshow(img_f)
            
            img = img_f.copy()
            
            string_actions = string_actions + 'f'
            
            ## img_pil_cf  = Image.fromarray(img_i_cen_crop_r)
            
        if (DO_ROTATE):
            
            # Look up the rotation angle
            
            dt_since_start = (t_arr[i] - t_arr[0]).total_seconds()
            delta_angle = angleField_arr[int(dt_since_start)]
    
            # De-rotate this image
    
            img_r = ndimage.rotate(img, delta_angle*r2d, reshape=False) 
                        
            print(f'Rotated image at t={dt_since_start} sec by {delta_angle * r2d} deg')

            string_actions = string_actions + 'r'

            # Stack the original and rotated together
            
            img_r_stack = np.hstack((img, img_r))
            # img_pil_2 = Image.fromarray(img_out_2)
            plt.imshow(img_r_stack)
            plt.show()
        
        # Save the image
        # PNG allows a 16-bit unsigned int, so use that!        
        # file_out   = file.replace(path_in, path_out).replace('.fit', '.png')
        
        # img_out     = img.copy() * ((2**16))/np.amax(img) ## Scaling is weird here. Not sure why I need to do this.
        
        img_out = (img.copy() - np.amin(img)).astype("uint16")
        
        img_pil_out = Image.fromarray(img_out)
        # if img_pil_out.mode != 'RGB':
          # img_pil_out = img_pil_out.convert('I')
        
        file_out   = file.replace(path_in, path_out).replace('.fit', '_' + string_actions + '.png')
        
        img_pil_out.save(file_out)
        print(f'{i}/{len(files)}: Wrote: ' + file_out)
        
        plt.show()

        # print("\n")
        

def getDiskFlattened(img, mask, x, y):  # Return an array, which is a cleaned version of the image

 # Do a fit to the disk itself. We want to flatten the disk, essentially.
    
    p_init = models.Polynomial2D(degree=1)
 
    mask = getMaskDisk(img)
    
    fit_p = fitting.LinearLSQFitter()
    p_disk = fit_p(p_init, x[mask], y[mask], img[mask])

    img_flattened = img.copy()
    img_flattened2 = img.copy()
    
    # Subtract the gradient off of just the disk
    # This works OK, but there's occasionally a hard transition right at the edge of the sun.
    
    img_flattened[mask] = img_flattened[mask] - p_disk(x[mask],y[mask]) + np.mean( p_disk(x[mask],y[mask]) )
    
    # Or, substract the gradient off of the entire image (disk + bg)
    # This is not great, since it sets the background level very much off.
    # I think it would be better if it was a soft edge.
    
    # In reality what we want is both of these combined. Some of the signal is truly a linear haze to subtract,
    # and some of the signal is truly a gradient on the disk (caused by etalon).
    
    # Bug: we have some wraparound issues here to be fixed. Highs are clipped.
    
    img_flattened2 = img_flattened2 - (p_disk(x,y)) + (np.mean(p_disk(x[mask],y[mask])))
    img_flattened2 -= np.amin(img_flattened2)
    img_flattened2 = img_flattened2.astype('uint16')
    
    # Just for r
    img_flattened3 = (img_flattened/2 + img_flattened2/2).astype('uint16')
    
    return(img_flattened3)


    
def test():

      ## This file has DSII with no polarizer
      
    file = '/Users/throop/Data/Solar/Movie_17Mar23/Light_Sun_10.0ms_Bin1_20230317-120747_0011.fit'

    ## DSII w/ polarizer
    
    file = '/Users/throop/Data/Solar/Movie_17Mar23/Light_Sun_30.0ms_Bin1_20230317-163358_0060.fit'
    
    hdu = fits.open(file)
    img = hdu['PRIMARY'].data
    header = hdu['PRIMARY'].header
    date_obs = header['DATE-OBS']
    t = datetime.strptime(date_obs,"%Y-%m-%dT%H:%M:%S.%f")
    hdu.close()

    isDisk = getMaskDisk(img)
    isDisk_s = getMaskDisk(img_s)
    
    isDisk = img > np.percentile(img,95)/4 # Flag the solar disk this way

    factor_s = 10  # How much to smallen each dimension by
    
    img_s = img[::factor_s, ::factor_s]
    kernel = 1 + np.zeros((10,10))
    kernel = kernel / np.sum(kernel)
    isDisk_bigger_s = ndimage.convolve(isDisk_s, kernel, mode='constant', cval=0.0) > 0
    isDisk_smaller_s = (isDisk_s.astype(float) - 
                        (ndimage.convolve(isDisk_s, kernel, mode='constant', cval=0.0))) < 0.001
    isDisk_smaller_s = np.logical_and(isDisk_smaller_s, isDisk_s)
    
    plt.imshow(isDisk_smaller_s)
    
    bg_s = img_s.copy().astype(float)
    bg_s[isDisk_bigger_s] = np.nan

    y, x = np.mgrid[:np.shape(img)[0], :np.shape(img)[1]]

    not_nans = np.isfinite(bg_s) # Get the list of finite values 
    p_init = models.Polynomial2D(degree=5)
    
    # p_init = models.Moffat2D()
    # p_init = models.RickerWavelet2D()
    # p_init = models.Ring2D()

    # Do a fit to the background
    
    fit_p = fitting.LinearLSQFitter()
    p_bg = fit_p(p_init, x[not_nans], y[not_nans], bg_s[not_nans])  # only fit the finite values
    norm=simple_norm(img_s-p_bg(x,y), percent=99)
    plt.imshow(img_s - p_bg(x,y),norm=norm)
    plt.title('background flattened')
    plt.show()
    
    # Do a fit to the solar disk
    
    p_init = models.Polynomial2D(degree=2)    
    p_disk = fit_p(p_init, x[isDisk_smaller_s], y[isDisk_smaller_s], img_s[isDisk_smaller_s])  # only fit the finite values
    plt.imshow(p_disk(x,y))

    sfit = p_disk(x,y)
    m = np.mean(img_s[isDisk_s])
    
    img_composite_s = img_s.copy().astype(float)
    # img_composite_s[isDisk_s] = img_composite_s[isDisk_s] - sfit[isDisk_smaller_s] + m
    img_composite_s[isDisk_s] = img_composite_s[isDisk_s] - sfit[isDisk_s] + m # this is correct
    
    # img_composite_s = img_s - sfit 
        
    norm=simple_norm(img_composite_s, max_percent=100, min_percent=85, stretch='linear')
    plt.imshow(img_composite_s,norm=norm)
    plt.title('solar flattened')
    plt.show()
    
    # Plot four images as output
    
    img_out_top = np.hstack((img_s, bg_s))
    img_out_bot = np.hstack((p(x,y), img_s-p(x,y)))
    img_out_merged = np.vstack((img_out_top,img_out_bot))
    plt.imshow(img_out_merged)  
    plt.show()


# Alternatively, fit the Sun itself

    # p_init = models.Polynomial2D(degree=3)
    # fit_p = fitting.LinearLSQFitter()

    p_init = models.Disk2D(amplitude=3000, x_0=250,y_0=200,R_0=50)
    fit_p  = fitting.LevMarLSQFitter()
    
    p = fit_p(p_init, x, y, img_s)  
    img_out_s2 = np.hstack((img_s, p(x,y)))
    plt.imshow(img_out_s2)  
    plt.show()
    
    print(fit_p.fit_info)

def fits2png():

    path_in = '/Volumes/Data/Solar/19Feb24'
    path_out = os.path.join(path_in, 'out')
    
    if not(os.path.exists(path_out)):
        os.mkdir(path_out)
        print('Create: ' + path_out)
        
    files = glob.glob(os.path.join(path_in, '*.fit*'))
    files = np.sort(files)
        
    plt.set_cmap(plt.get_cmap('Greys_r'))

    for i,file in enumerate(files):

        hdu = fits.open(file)
        img = hdu['PRIMARY'].data
        header = hdu['PRIMARY'].header
        date_obs = header['DATE-OBS']
        t = datetime.strptime(date_obs,"%Y-%m-%dT%H:%M:%S.%f")
        hdu.close()    
        
        # img_c = img[1000:3500, 500:3000]
        img_c = img[500:3000, 1000:3500]

        plt.imshow(img_c)
        plt.show()
    
        file_out = file.replace(path_in, path_out).replace( '.fit', '.png')
        img_pil   = Image.fromarray(img_c)
        img_pil.save(file_out)

        print(f'{i}/{len(files)}: Wrote: ' + file_out)


if (__name__ == '__main__'):
    process_all()        

