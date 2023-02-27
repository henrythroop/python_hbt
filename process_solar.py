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
import spiceypy as sp
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

import scipy.misc

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

def derotate():
    
# SPICE logic to calc az, elev for Sun
#     RECAZEL
# Q: I can get the vector in J2000 space.
# I need to get it relative to 
# Maybe I just n3ed to define a new reference frame, for Colombo?
# And then get vector from Earth to Sun, in that frame?

    d2r = 2*math.pi/360
    r2d = 1/d2r
    
    ut_start = "2023-02-13T07:28:35.594129"
    ut_end   = "2023-02-13T12:28:35.594129"
    
    file_tm = 'kernels_base.tm'
    sp.unload(file_tm)
    sp.furnsh(file_tm)
    
    et_start = sp.utc2et(ut_start)
    et_end = sp.utc2et(ut_end)

    et = et_start
    
    lat_earth = 6*d2r
    lon_earth = 110*d2r # Check this
    
    code_earth = sp.bodn2c('Earth')
    
    pos_obs = sp.latrec(6700, lat_earth, lon_earth)
    pos_obs = sp.srfrec(code_earth, lat_earth, lon_earth)
    
    (st,lt) = sp.spkezr('Sun', et, 'IAU_EARTH', 'lt', 'Earth')
    (range, az, el) = sp.recazl(st[0:3], True, True)
    
def process_all():
    
    ## Define the size of the output image, in pixels
    
    sizeXOut = 2000
    sizeYOut = 2000
    
    DO_CROP = True
    DO_CENTROID = True
    DO_DEROTATE = False  # Either use SPICE, or https://github.com/cytan299/field_derotator/tree/master/field_derotator_formula
    DO_LIMBFIT = False
    
    
    # Set the path of the files to look at
    
    path_in = '/Users/throop/Data/Solar/Movie_16Feb23'
    path_out = os.path.join(path_in, 'out')
    if not(os.path.exists(path_out)):
        os.mkdir(path_out)
        print('Create: ' + path_out)
        
    files = glob.glob(os.path.join(path_in, '*.fit*'))
    files = np.sort(files)
    
    file = files[0]
    
    plt.set_cmap(plt.get_cmap('Greys_r'))

    for file in files:
        
        # Read the original data + header
        
        hdu = fits.open(file)
        img = hdu['PRIMARY'].data
        header = hdu['PRIMARY'].header
        hdu.close()
        
        img_out = np.zeros((sizeXOut,sizeYOut), dtype='uint16')
        
        isDisk = img > np.max(img)/10  # This seems to flag the solar disk
        
        # Find center-of-mass, X dir

        (centerY, centerX) = scipy.ndimage.measurements.center_of_mass(isDisk)
        (widthY, widthX) = np.shape(isDisk)
        
        # Create a new image, based on this centering
        
        img_out = img[int(widthY/2 + int(centerY-widthY/2) - sizeYOut/2):
                      int(widthY/2 + int(centerY-widthY/2) + sizeYOut/2),
                      int(widthX/2 + int(centerX-widthX/2) - sizeXOut/2):
                      int(widthX/2 + int(centerX-widthX/2) + sizeXOut/2)]
        
        plt.imshow(img_out)                            
        
        plt.show()
        
        # Save the image
        # Looks like PNG allows a 16-bit unsighed int, so use that!
        
        image_pil = Image.fromarray(img_out)
        file_out = file.replace(path_in, path_out).replace('.fit', '.png')
        image_pil.save(file_out)
        print('Wrote: ' + file_out)
        
        

        # if cell > 0:
        #     file_out = os.path.join(path_out,
        #        os.path.basename(file).replace('.fits', f'_{cell}.fits'))
        # else:
        #     file_out = os.path.join(path_out, os.path.basename(file))
            
        # Write the new FITS file
        
        # hdu_out = fits.PrimaryHDU(img_d, header=header)
        # hdu_out.writeto(file_out, output_verify='ignore', overwrite=True)
        # print(f'Wrote: {file_out}')
        

if (__name__ == '__main__'):
    process_all()        

