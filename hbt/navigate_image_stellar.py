#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 16:15:18 2017

@author: throop
"""

"""
Library routine to navigate an image based on stars.
  o Use DAOPhot to locate stars in the image
  o Use a star catalog to look up stars that should be in the image
  o Calculate the offset between these.
  o Return the result, in terms of a revised WCS for the frame
"""

# General python imports

import pdb
import glob
import math       # We use this to get pi. Documentation says math is 'always available' 
                  # but apparently it still must be imported.
from   subprocess import call
import warnings
import pdb
import os.path
import os
import subprocess

import astropy
from   astropy.io import fits
from   astropy.table import Table
import astropy.table   # I need the unique() function here. Why is in in table and not Table??
import matplotlib.pyplot as plt # pyplot
from   matplotlib.figure import Figure
import numpy as np
import astropy.modeling
from   astropy.utils import data

from   scipy.optimize import curve_fit
                       # Pylab defines the 'plot' command
import spiceypy as sp
from   astropy.wcs import WCS
from   astropy.vo.client import conesearch # Virtual Observatory, ie star catalogs
from   astropy import units as u           # Units library
from   astropy.coordinates import SkyCoord # To define coordinates to use in star search
from   scipy.stats import mode
from   scipy.stats import linregress
import wcsaxes
import time
from   scipy.interpolate import griddata
import cv2

import re # Regexp
import pickle # For load/save

import cProfile # For profiling

from   matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from   matplotlib.figure import Figure
import warnings
from   importlib import reload

# HBT imports

import hbt

def navigate_image_stellar(im, wcs_in, name_catalog='GSC', DO_PLOT=True):

# Inputs are the image array, and the WCS structure.
# This routine does not do any file IO. The image array and header must be already loaded.
# The image is assumed to be stretched properly s.t. stars can be found using DAOphot. 

    NUM_STARS_PHOT = 100
    NUM_STARS_CAT  = 100

#==============================================================================
# Calculate the image radius, in radians, based on the size and the pixel scale
#==============================================================================

    dx_pix = hbt.sizex(im)
    dy_pix = hbt.sizey(im)
    radec_corner = wcs_in.wcs_pix2world(0, dy_pix/2, 0)
    radec_center = wcs_in.wcs_pix2world(dx_pix/2, dy_pix/2, 0)
    (ra_corner, dec_corner) = radec_corner
    (ra_center, dec_center) = radec_center
    
    radius_image = math.sqrt((dec_corner-dec_center)**2 + 
                             ((ra_corner-ra_center) / np.cos(dec_corner*hbt.d2r))**2) * hbt.d2r

    radius_search_deg = radius_image * hbt.r2d
    
# Load the image

#hdulist  = fits.open(file)
#image    = hdulist['PRIMARY'].data
#header   = hdulist['PRIMARY'].header

# Read the WCS coordinates

#with warnings.catch_warnings():
#    warnings.simplefilter("ignore")    
#    w = WCS(file)                # Look up the WCS coordinates for this frame
#                                 # Otherwise it gives "FITSFixedWarning: 'unitfix': 'Changed units: 'DEG' -> 'deg'"
# Read the WCS parameters
           
    center_deg  = wcs_in.wcs.crval  # degrees. # crval is a two-element array of [RA, Dec], in degrees

# Initialize SPICE

#sp.furnsh(file_tm)
#et = header['SPCSCET']
#utc = sp.et2utc(et, 'C', 1)

# Stretch the image

    stretch_percent = 90
    stretch = astropy.visualization.PercentileInterval(stretch_percent)  # PI(90) scales array to 5th .. 95th %ile. 

# Display it

    if (DO_PLOT):
        plt.imshow(stretch(im))
    
    # Load matching stars
    
    DO_GSC1     = False    # Stopped working 2-Oct-2016
    DO_GSC2     = True
    DO_USNOA2   = False

#==============================================================================
# Get stars from star catalogs     
#==============================================================================

    if (DO_GSC1):
        name_cat = u'The HST Guide Star Catalog, Version 1.1 (Lasker+ 1992) 1' # works, but 1' errors; investigating
        stars = conesearch.conesearch(center_deg, radius_search_deg, cache=False, catalog_db = name_cat)
        ra_stars  = np.array(stars.array['RAJ2000'])*hbt.d2r # Convert to radians
        dec_stars = np.array(stars.array['DEJ2000'])*hbt.d2r # Convert to radians
    #            table_stars = Table(stars.array.data)
    
    if (DO_GSC2):
        name_cat = u'Guide Star Catalog v2 1'
            
        with data.conf.set_temp('remote_timeout', 30): # This is the very strange syntax to set a timeout delay.
                                                       # The default is 3 seconds, and that times out often.
            stars = conesearch.conesearch(wcs_in.wcs.crval, radius_search_deg, cache=True, catalog_db = name_cat)
    
        ra_stars  = np.array(stars.array['ra'])*hbt.d2r # Convert to radians
        dec_stars = np.array(stars.array['dec'])*hbt.d2r # Convert to radians
    
        mag       = np.array(stars.array['Mag'])
        
        print("Stars downloaded: N = {}; mag = {:.2f} .. {:.2f}".format(np.size(mag), np.nanmin(mag), np.nanmax(mag)))
        print("RA = {:.2f} .. {:.2f}".format(np.nanmin(ra_stars)*hbt.r2d, np.nanmax(ra_stars)*hbt.r2d))
        
        # Now sort by magnitude, and keep the 100 brightest
        # This is because this GSC catalog is huge -- typically 2000 stars in LORRI FOV.
        # We need to reduce its size to fit in our fixed astropy table string length.
    
        order = np.argsort(mag)
        order = np.array(order)[0:NUM_STARS_CAT]
    
        ra_stars = ra_stars[order]   # Returned as radians
        dec_stars = dec_stars[order]
    
    if (DO_USNOA2):  
        name_cat = u'The USNO-A2.0 Catalogue (Monet+ 1998) 1' # Works but gives stars down to v=17; I want to v=13 
        stars = conesearch.conesearch(wcs_in.wcs.crval, 0.3, cache=False, catalog_db = name_cat)
        table_stars = Table(stars.array.data)
        mask = table_stars['Bmag'] < 13
        table_stars_m = table_stars[mask]            
    
        ra_stars  = table_stars_m['RAJ2000']*hbt.d2r # Convert to radians
        dec_stars = table_stars_m['DEJ2000']*hbt.d2r # Convert to radians
    
    ra_stars_cat  = ra_stars
    dec_stars_cat = dec_stars

    radec_stars_cat        = np.transpose(np.array((ra_stars_cat, dec_stars_cat)))
    
    (x_stars_cat, y_stars_cat) = wcs_in.wcs_world2pix(
                                                      radec_stars_cat[:,0]*hbt.r2d, 
                                                      radec_stars_cat[:,1]*hbt.r2d, 0)   
    
    points_stars_cat = np.transpose((y_stars_cat, x_stars_cat))  # Yes, order is supposed to be (y,x)
    
#==============================================================================
# Adjust for stellar aberration, if requested
#==============================================================================

    DO_ABCORR = False

    if DO_ABCORR:
    
        # Look up velocity of NH, for stellar aberration
        
        abcorr = 'LT+S'
        frame = 'J2000'
        st,ltime = sp.spkezr('New Horizons', et, frame, abcorr, 'Sun') # Get velocity of NH 
        vel_sun_nh_j2k = st[3:6]
        
        # Correct stellar RA/Dec for stellar aberration
        
        radec_stars_cat_abcorr = hbt.correct_stellab(radec_stars_cat, vel_sun_nh_j2k) # Store as radians
        
        x_stars_cat_abcorr, y_stars_cat_abcorr   = w.wcs_world2pix(radec_stars_cat_abcorr[:,0]*r2d, 
                                                                   radec_stars_cat_abcorr[:,1]*r2d, 0)
    
        points_stars_cat_abcorr = np.transpose((y_stars_cat_abcorr, x_stars_cat_abcorr))

#==============================================================================
# Use DAOphot to search the image for stars.
#==============================================================================
  
    points_stars_phot = hbt.find_stars(im, num=NUM_STARS_PHOT) # Returns N x 2 aray. 0 = Row = y; 1 = Column = x.
    
    y_stars_phot =(points_stars_phot[:,0]) # xy is correct -- see above
    x_stars_phot =(points_stars_phot[:,1]) # 

#==============================================================================
# Make a plot showing the DAO stars on the image
#==============================================================================

    color_usno = 'red'
    
    DO_PLOT_DAO = True
    
    if (DO_PLOT_DAO):

        plt.imshow(stretch(im))

        plt.plot(x_stars_phot, y_stars_phot, linestyle='none', 
                 marker='o', markersize=9, mec=color_usno, mew=1, color='none', 
                 label = 'DAO photometric stars') # plot() uses x, y

        plt.plot(x_stars_cat, y_stars_cat, linestyle='none', 
                 marker='o', markersize=5, color='lightgreen', 
                 label = 'Cat stars') # plot() uses x, y        

        plt.ylim((1024,0))
        plt.xlim((0,1024))
        plt.legend(loc = 'upper left')
        plt.show()

# Up til here, x and y are correct
    
#==============================================================================
# Look up the shift between the photometry and the star catalog. 
# Do this by making a pair of fake images, and then looking up image registration on them.
#==============================================================================

# I call this pointing process 'opnav'. 
# It is returned in order (y,x) because that is what imreg_dft uses, even though it is a bit weird.
 
    # Use the ICP / CV2 method to do this, which is better than my own custom method
    
#    (popt, mat) = icp(np.transpose(points_stars_phot), np.transpose(points_stars_cat))
    (popt, mat) = icp(np.transpose(points_stars_cat),  np.transpose(points_stars_phot))

    dy_opnav_cv2 = mat[1,2] # swapping
    dx_opnav_cv2 = -mat[0,2]
    
    dx_opnav = dx_opnav_cv2
    dy_opnav = dy_opnav_cv2
    
#    (dy_opnav, dx_opnav) = hbt.calc_offset_points(points_stars_phot, points_stars_cat, np.shape(im),
#        diam_kernel=9, labels = ['DAO', 'GSC Catalog'], do_plot_before=True, do_plot_after=True, 
#        do_plot_raw = True, do_binary = True)

    dy = dy_opnav
    dx = dx_opnav

#    points_stars_cat_xform = cv2.transform( np.array([points_stars_cat.astype(np.float32) ]), np.array(mat[0:2]))
    points_stars_cat_xform = cv2.transform( np.array([points_stars_cat.astype(np.float32) ]), np.array(mat))
    
    print("CV2: dx = {}, dy = {}".format(dx_opnav_cv2, dy_opnav_cv2))

# Try new algorithms

    ((dx_opnav_2, dy_opnav_2), mat_2) =  get_translation_points(points_stars_cat, points_stars_phot)
    ((dx_opnav_2_r, dy_opnav_2_r), mat_2_r) =  get_translation_points(points_stars_phot, points_stars_cat)

    diam_kernel = 9
    do_binary = True
    
    image_cat  = image_from_list_points(points_stars_cat,  shape, diam_kernel, do_binary=do_binary)
    image_phot = image_from_list_points(points_stars_phot, shape, diam_kernel, do_binary=do_binary)
    
    im_a = image_cat
    im_b = image_phot
    
#    ((dx_opnav_3, dy_opnav_3), mat_3)       = get_translation_images(image_cat, image_phot)
#    ((dx_opnav_3_r, dy_opnav_3_r), mat_3_r) = get_translation_images(image_phot, image_cat)

    (dy_opnav_4, dx_opnav_4) = ird.translation(image_cat, image_phot)['tvec']
            
    dy_opnav = -dy_opnav_4
    dx_opnav = -dx_opnav_4
    
    print("Opnav_2: Points")
    print("Opnav_2:   dx={}, dy={}".format(dx_opnav_2,   dy_opnav_2))
    print("Opnav_2_r: dx={}, dy={}".format(dx_opnav_2_r, dy_opnav_2_r))
    
    print("Opnav_3: Images")
    print("Opnav_3:   dx={}, dy={}".format(dx_opnav_3,   dy_opnav_3))
#    print("Opnav_3_r: dx={}, dy={}".format(dx_opnav_3_r, dy_opnav_3_r))

    print("Opnav_4: Images, FFT")
    print("Opnav_4:   dx={:.2f}, dy={:.2f}".format(dx_opnav_4,   dy_opnav_4))
    
    # Conclusion: Method 3 (get_translation_images) works better than Method 2
    
#==============================================================================
# Make a plot, showing DAO positions + catalog positions
#==============================================================================

    if (DO_PLOT):
        
        hbt.figsize((10,10))
        
        plt.imshow(stretch(im))
        
        # Plot the stars -- catalog, and DAO
        
        plt.plot(x_stars_cat + dx_opnav, y_stars_cat + dy_opnav, 
                 marker='o', ls='None', 
                 color='lightgreen', alpha = 0.5, ms=12, mew=1, label = 'Cat Stars, adjusted')

#        plt.plot(points_stars_cat_xform[0][:,1], points_stars_cat_xform[0][:,0], 
#                 marker='o', ls='None', 
#                 color='yellow', alpha = 0.7, ms=12, mew=1, label = 'Cat Stars, adjusted CV2')
        
        plt.plot(x_stars_cat, y_stars_cat, 
                 marker='o', ls='None', 
                 color='lightgreen', alpha = 1, ms=4, mew=1, label = 'Cat Stars, raw')
                 
        plt.plot(x_stars_phot, y_stars_phot, 
                 marker='o', ls='None', 
                 color='none', markersize=10, mew=1, mec='red', alpha = 1, label = 'DAOfind Stars')               
        
    
        plt.title('After navigation, with dx = {:.1f}, dy = {:.1f}'.format(dx_opnav, dy_opnav))
        plt.legend()  # Draw legend. Might be irrel since remove() might keep it; not sure.
        
        plt.imshow(stretch(im))
        plt.show

#==============================================================================
# Return results and exit
#==============================================================================

    return(dx_opnav, dy_opnav)    

#==============================================================================
# Routine for testing only
#==============================================================================

def TESTING():

    dir =  '/Users/throop/Dropbox/Data/NH_Jring/data/jupiter/level2/lor/all/' 
    file = dir + 'lor_0034765323_0x630_sci_1.fit' # This one is faint -- hard to see much. But algo works -both.
    file = dir + 'lor_0034602123_0x630_sci_1.fit'  # Algo works. Both g_t_i and fft.
    file = dir + 'lor_0034613523_0x630_sci_1.fit'  # Algo works on both types.
    file = dir + 'lor_0034676528_0x630_sci_1.fit' # Good test case, very simple, good stars. g_t_i works great.

    im = hbt.read_lorri(file)
    
    DO_PLOT = True
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")    
        w = WCS(file)                # Look up the WCS coordinates for this frame
    
    plt.set_cmap('Greys_r')
    hbt.figsize((10,10))
    
    wcs_in = w
    
    out = navigate_image_stellar(im, w)

#==============================================================================
# Function to get the translation between two sets of points
#==============================================================================

def get_translation_points(a, b): # a, b have shape (N, 2).
                                  # a[0] is the first ordered pair, etc.  

    # Break apart into sub-arrays
    
    a_x = np.array(a[:,1])
    a_y = np.array(a[:,0])
    b_x = np.array(b[:,1])
    b_y = np.array(b[:,0])
    
    shift_max = 30 
    INVALID = -999
    
    range_dy = np.arange(-shift_max, shift_max+1)  # Actually goes from -N .. +(N-1)
    range_dx = np.arange(-shift_max, shift_max+1)
        
    sum = np.zeros((hbt.sizex(range_dx), hbt.sizex(range_dy))) # Create the output array. 
                                         # Each element here is the sum of (dist from star N to closest shifted star)
    num_valid = 0
    
    for j,dx in enumerate(range_dx):
        for k,dy in enumerate(range_dy):
            sum_i = 0
            num_valid = 0
            for a_i in a:
                a_i_shift = (a_i[1] + dy, a_i[0] + dx)
                a_i_shift_x = a_i_shift[1]
                a_i_shift_y = a_i_shift[0]
                dist_i = np.sqrt((a_i_shift_x - b_x)**2 + (a_i_shift_y - b_y)**2) # dist btwn a_i and all b's
                
                if (np.amin(dist_i) < shift_max/2):
                    sum_i += np.amin(dist_i)
                    num_valid += 1
            if (num_valid > 0):
                sum[j,k] = sum_i/num_valid
            else:
                sum[j,k] = INVALID
            
    sum[sum == INVALID] = np.median(sum[np.where(sum != INVALID)])
    
    pos = np.where(sum == np.amin(sum)) # Assumes a single valued minimum
    
    dy_out = pos[1][0] - shift_max
    dx_out = pos[0][0] - shift_max
        
    return (np.array((dy_out, dx_out)), sum)

#==============================================================================
# Function to get the translation between two sets of points
#==============================================================================

def get_translation_images(im_a, im_b):

    # out = get_translation_images(image_cat, np.roll(np.roll(image_cat, 10, 0), -7, 1))
            
    shift_max = 50 # Shift halfwidth
    
    range_dy = np.arange(-shift_max, shift_max+1)  # Actually goes from -N .. +(N-1)
    range_dx = np.arange(-shift_max, shift_max+1)
        
    sum = np.zeros((hbt.sizex(range_dx), hbt.sizex(range_dy))) # Create the output array. 
                                         # Each element here is the sum of (dist from star N to closest shifted star)
    
    for j,dx in enumerate(range_dx):
        for k,dy in enumerate(range_dy):
            sum[j,k] = np.sum(np.logical_and( im_a, np.roll(np.roll(im_b,dx,0),dy,1)) )
        print("j = {}, dx = {}".format(j, dx))
        
    pos = np.where(sum == np.amax(sum)) # Assumes a single valued minimum
    
    dy_out = pos[1][0] - shift_max
    dx_out = pos[0][0] - shift_max
        
    return (np.array((dy_out, dx_out)), sum)
                