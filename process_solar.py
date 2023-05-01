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
# import photutils
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
    
    
    lon = 79.853        
    lat = 6.913
    
    result = get_position(date, lon, lat)
    az = result['azimuth']
    alt = result['altitude']
    print('Az = ' + str(az * r2d) + '; Alt = ' + str(alt * r2d))
    
    date = datetime.now()
    
    delta = timedelta(seconds=3600*5.5)

    # Verify that sunrise and sunset elevations are correct. Yes they are -- but I need to pass local time, not UTC!
    # This is probably an issue with the timedate() module, moreso than with this code.
    
    get_position(get_times(date, lon, lat)['sunset'] + timedelta(seconds=3600*5.5), lon, lat)
    get_position(get_times(date, lon, lat)['sunrise'] + timedelta(seconds=3600*5.5), lon, lat)
    
    get_times(datetime.now(), lon, lat)
    
def process_all():
    
    ## Define the size of the output image, in pixels
    
    sizeXOut = 1600 # This is arbitrary
    sizeYOut = 1600
    
    sizeXOut = 2800 # This is arbitrary
    sizeYOut = 2800
    
    DO_CROP     = True
    DO_CENTROID = True
    DO_DEROTATE = False  # Either use SPICE, or https://github.com/cytan299/field_derotator/tree/master/field_derotator_formula
    DO_LIMBFIT  = False

    d2r = 2*math.pi/360
    r2d = 1/d2r
#     omegaEarth = 2*math.pi / 86400  # Earth rotation rate, radians/sec.
    omegaEarth = 4.178e-3 * math.pi / 180 ## Rad/sec, fixed

    lon = 79.853        
    lat = 6.913
    
    # Set the path of the files to look at
    
    path_in = '/Volumes/Data/Solar/Eclipse_20Apr23'
    path_out = os.path.join(path_in, 'out')
    if not(os.path.exists(path_out)):
        os.mkdir(path_out)
        print('Create: ' + path_out)
        
    files = glob.glob(os.path.join(path_in, '*.fit*'))
    files = np.sort(files)
    
    file = files[0]
    
    plt.set_cmap(plt.get_cmap('Greys_r'))

# Do a first pass through the files, to get start and end times

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
      
      fudge = 3600*5.5 # For some reason, need to do a timezone offset. Not sure why.
      
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
      
# Now do the processing of each image

    for i,file in enumerate(files[145:]):
        
        # Read the data + header
        
        hdu = fits.open(file)
        img = hdu['PRIMARY'].data
        header = hdu['PRIMARY'].header
        date_obs = header['DATE-OBS']
        t = datetime.strptime(date_obs,"%Y-%m-%dT%H:%M:%S.%f")
        hdu.close()
        
        img_out = np.zeros((sizeXOut,sizeYOut), dtype='uint16')
        
        isDisk = getMaskDisk(img)
        
        # Find center-of-mass, X dir

        (centerY, centerX) = scipy.ndimage.measurements.center_of_mass(isDisk)
        (widthY, widthX) = np.shape(isDisk)
        
        # Create a new image, based on this centering
        
        img_out = img[int(widthY/2 + int(centerY-widthY/2) - sizeYOut/2):
                      int(widthY/2 + int(centerY-widthY/2) + sizeYOut/2),
                      int(widthX/2 + int(centerX-widthX/2) - sizeXOut/2):
                      int(widthX/2 + int(centerX-widthX/2) + sizeXOut/2)]

        # Error checking: If we get too close to the edge, the array returnd will be too small. 
        # If this happens, no error is thrown. So we check, and if so, we pad the array fatly, and try again
        
        if np.shape(img_out) != (widthY, widthX):
            
            dpad = 500  # Extra amount to pad each edge by
            
            img_pad = np.pad(img, ((dpad,dpad), (dpad,dpad)), mode='constant', constant_values=np.median(img))
            
            img_out = img_pad[int(widthY/2 + dpad + int(centerY-widthY/2) - sizeYOut/2):
                          int(widthY/2 + dpad + int(centerY-widthY/2) + sizeYOut/2),
                          int(widthX/2 + dpad + int(centerX-widthX/2) - sizeXOut/2):
                          int(widthX/2 + dpad + int(centerX-widthX/2) + sizeXOut/2)]
            
        if (i==0):  # Do this only the first time through            
          y, x = np.mgrid[:np.shape(img_out)[0], :np.shape(img_out)[1]]

        # Remove a linear gradiant from the image
        # Add a suffix '_f', for 'flattened'
    
        mask = getMaskDisk(img_out)
        img_out_f = getDiskFlattened(img_out, mask, x, y)                     
        
        # Look up the rotation angle
        
        dt_since_start = (t - t_0).total_seconds()
        delta_angle = angleField_arr[int(dt_since_start)]

        # De-rotate this image
        # Add suffix _r, for 'rotate'

        img_pil   = Image.fromarray(img_out_f)
        # img_out_rf = ndimage.rotate(img_pil, delta_angle*r2d, reshape=False) 
        img_out_r = ndimage.rotate(img_out, delta_angle*r2d, reshape=False) 
        
        # img_pil_rf = Image.fromarray(img_out_rf)
        img_pil_r  = Image.fromarray(img_out_r)
        
        print(f'Rotated image at t={dt_since_start} sec by {delta_angle * r2d} deg')

        # Stack the original and processed together
        
        img_out_2 = np.hstack((img_out, img_out_r))
        # img_pil_2 = Image.fromarray(img_out_2)
        plt.imshow(img_out_2)
        plt.show()
        
        # Save 
        # Save the image
        # PNG allows a 16-bit unsigned int, so use that!        
        # file_out   = file.replace(path_in, path_out).replace('.fit', '.png')
        
        file_out_rf = file.replace(path_in, path_out).replace('.fit', '_rf.png')
        file_out_r = file.replace(path_in, path_out).replace( '.fit', '_r.png')
    
        # img_pil.save(file_out)
        # img_pil_rf.save(file_out_rf)
        # print(f'{i}/{len(files)}: Wrote: ' + file_out_rf)        
        
        img_pil_r.save(file_out_r)
        print(f'{i}/{len(files)}: Wrote: ' + file_out_r)

        # file_out_2 = file.replace(path_in, path_out).replace('.fit', '_2.png')
        # img_pil_2.save(file_out_2)
        # print(f'{i}/{len(files)}: Wrote: ' + file_out_2)
        print("\n")

def getMaskDisk(img): # Return a mask which is the solar disk itself

    isDisk = img > np.percentile(img, 95)/4
    
    return(isDisk)


def getDiskFlattened(img, mask, x, y):  # Return an array, which is a fit to the solar disk

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
    # There is some wraparound error, causing a white-black transition beyond the edge.
    # I think it would be better if it was a soft edge.
    
    img_flattened2 = img_flattened2 - (p_disk(x,y)).astype('uint16') + (np.mean(p_disk(x[mask],y[mask]))).astype('uint16')
    
    return(img_flattened2)

    
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

    path_in = '/Volumes/Data/Solar/Eclipse_20Apr23'
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

