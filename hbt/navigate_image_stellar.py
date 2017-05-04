#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 16:15:18 2017

@author: throop
"""

"""
Library function to navigate an image based on stars.
  o Use DAOPhot to locate stars in the image
  o Use a star catalog to look up stars that should be in the image
  o Calculate the offset between these.
  o Return the result, in terms of a revised WCS for the frame, and a pixel offset.
  o Does not rewrite the FITS header.
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
#import cv2

import re # Regexp
import pickle # For load/save

import cProfile # For profiling

from   matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from   matplotlib.figure import Figure
import warnings
from   importlib import reload
from   time import gmtime, strftime

# HBT imports

import hbt

def navigate_image_stellar(im, wcs_in, name_catalog='', do_plot=True, method='fft', title=''):

    """
    Navigate frame based on stellar images.
    Result returns is pixel shift (dy, dx).
    WCS paramaters are returned, *and* modified in place.
    """
    
    import imreg_dft as ird
    from   astropy.wcs import WCS
    from   astropy.vo.client import conesearch # Virtual Observatory, ie star catalogs
  
# Inputs are the image array, and the WCS structure.
# This routine does not do any file IO. The image array and header must be already loaded.
# The image is assumed to be stretched properly s.t. stars can be found using DAOphot. 

    NUM_STARS_PHOT = 100  # How many stars to use from DAOPhot. For noisy images, DAO will find a lot of
                          # fake stars, so we need to crank this up higher than the # of cat stars.
    NUM_STARS_CAT  = 50  # How many stars to use from star catalog

    DO_GSC1     = False
    DO_GSC12     = True
    DO_USNOA2   = False
    
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
    
# Read the WCS coordinates
           
    center_deg  = wcs_in.wcs.crval  # degrees. # crval is a two-element array of [RA, Dec], in degrees

# Stretch the image. This is just for display -- no processing.

    stretch_percent = 90
    stretch = astropy.visualization.PercentileInterval(stretch_percent)  # PI(90) scales array to 5th .. 95th %ile. 

# Display it

    if (do_plot):
        plt.imshow(stretch(im))

#==============================================================================
# Get stars from star catalogs     
#==============================================================================
    
    if (DO_GSC1):
        name_cat = u'The HST Guide Star Catalog, Version 1.1 (Lasker+ 1992) 1' # works, but 1' errors; investigating
        stars = conesearch.conesearch(center_deg, radius_search_deg, cache=True, catalog_db = name_cat)
        ra_stars  = np.array(stars.array['RAJ2000'])*hbt.d2r # Convert to radians
        dec_stars = np.array(stars.array['DEJ2000'])*hbt.d2r # Convert to radians
    #            table_stars = Table(stars.array.data)
    
    if (DO_GSC12):
#        name_cat = u'The HST Guide Star Catalog, Version 1.2 (Lasker+ 1996) 1'
        name_cat = u'Guide Star Catalog v2 1'                                       # Works from gobi, not tomato
        url_cat = 'http://gsss.stsci.edu/webservices/vo/ConeSearch.aspx?CAT=GSC23&' # Works always
            
        with data.conf.set_temp('remote_timeout', 30): # This is the very strange syntax to set a timeout delay.
                                                       # The default is 3 seconds, and that times out often.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")                                                          
                stars = conesearch.conesearch(wcs_in.wcs.crval, radius_search_deg, cache=True, catalog_db = url_cat)
    
        ra_stars  = np.array(stars.array['ra'])*hbt.d2r  # Convert to radians
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
# Use DAOphot to search the image for stars.
#==============================================================================
  
    points_stars_phot = hbt.find_stars(im, num=NUM_STARS_PHOT) # Returns N x 2 aray. 0 = Row = y; 1 = Column = x.
    
    y_stars_phot =(points_stars_phot[:,0]) # xy is correct -- see above
    x_stars_phot =(points_stars_phot[:,1]) # 

#==============================================================================
# Make a plot showing the DAO stars on the image
#==============================================================================

    color_phot = 'red'            # Color for stars found photometrically
    color_cat  = 'lightgreen'     # Color for stars in catalog
    
    DO_PLOT_DAO = False   # Plot an intermediate result?
    
    if (DO_PLOT_DAO):

        plt.imshow(stretch(im))

        plt.plot(x_stars_phot, y_stars_phot, linestyle='none', 
                 marker='o', markersize=9, mec=color_cat, mew=1, color='none', 
                 label = 'DAO photometric stars') # plot() uses x, y

        plt.plot(x_stars_cat, y_stars_cat, linestyle='none', 
                 marker='o', markersize=5, color='lightgreen', 
                 label = 'Cat stars') # plot() uses x, y        

        plt.title(title)
        plt.ylim((hbt.sizey(im)),0)
        plt.xlim((0,hbt.sizex(im)))
        plt.legend(loc = 'upper left')
        plt.show()

# Up til here, x and y are correct
    
#==============================================================================
# Look up the shift between the photometry and the star catalog. 
# Do this by making a pair of fake images, and then looking up image registration on them.
#==============================================================================

# I call this pointing process 'opnav'. 
# It is returned in order (y,x) because that is what imreg_dft uses, even though it is a bit weird.
    
    diam_kernel = 11  # How many pixels across are our synthetic stellar images? Should be odd number. Not critical.
    do_binary = True  # For the stellar images, do a binary 1/0 (recommended), or a pixel distance?

    shape = np.shape(im)   # Set shape of output array
    
    image_cat  = hbt.image_from_list_points(points_stars_cat,  shape, diam_kernel, do_binary=do_binary)
    image_phot = hbt.image_from_list_points(points_stars_phot, shape, diam_kernel, do_binary=do_binary)

    if (method == 'fft'):         # Very fast method

        # Set up a constraint for the fit. It should be different for 1x1 and 4x4.
        # For 1x1, it works well to be 100 pixels.

        if (hbt.sizex(im) == 1024):    # For LORRI 1x1
            constraint_tx    = (0,100) # Mean and stdev. i.e., returned value will be within stdev of mean.
            constraint_ty    = (0,100) 
            
        if (hbt.sizex(im) == 256):   # For LORRI 4x4 
            constraint_tx    = (0,25) # Mean and stdev. i.e., returned value will be within stdev of mean.
            constraint_ty    = (0,25)  
            
        constraint_angle = 0    # With one value, it is a fixed constraint.
        
        constraints = {'tx' : constraint_tx, 'ty' : constraint_ty, 'angle' : constraint_angle}
        ird.translation(image_cat, image_phot, constraints=constraints)
        
        (dy, dx) = ird.translation(image_cat, image_phot, constraints=constraints)['tvec']         
        dy_opnav = -dy
        dx_opnav = -dx

    if (method == 'bruteforce'):  # Very slow method

        ((dx, dy), mat)       = hbt.get_translation_images_bruteforce(image_cat, image_phot)
        dx_opnav = -dx
        dy_opnav = -dy
        
#==============================================================================
# Make a plot, showing DAO positions + catalog positions
#==============================================================================

    do_plot = True
    if (do_plot):
        
#        hbt.figsize((10,10))
        
        plt.imshow(stretch(im))
        
        # Plot the stars -- catalog, and DAO
        
        plt.plot(x_stars_cat + dx_opnav, y_stars_cat + dy_opnav, 
                 marker='o', ls='None', 
                 color=color_cat, alpha = 0.5, ms=12, mew=1, label = 'Cat Stars, adjusted')
        
        plt.plot(x_stars_cat, y_stars_cat, 
                 marker='o', ls='None', 
                 color=color_cat, alpha = 1, ms=4, mew=1, label = 'Cat Stars, raw')
                 
        plt.plot(x_stars_phot, y_stars_phot, 
                 marker='o', ls='None', 
                 color='none', markersize=10, mew=1, mec=color_phot, alpha = 1, label = 'DAOfind Stars')               
        
        plt.title('After navigation, with dx = {:.1f}, dy = {:.1f}, {}'.format(dx_opnav, dy_opnav, title))
        plt.legend()  # Draw legend. Might be irrel since remove() might keep it; not sure.
        
        plt.imshow(stretch(im))
        plt.show()

#==============================================================================
# Return results and exit
#==============================================================================

# Results are returned in terms of pixel offset and a revised WCS structure.
# I don't seem to be able to copy a WCS structure, so I modify the one in place!

# Get the pixel location of the center position

    crpix = wcs_in.wcs.crpix  # Center position, in pixels, old
    
# Get the new RA, Dec center of the array. It is just the old location, plus the offset
    
    ORIGIN_FORMAT = 1  # 0 for Numpy-style indexing, 1 for Fortran-style and FITS-style.
                       # So what do I used for FITS files in python? Experimentally, 1 is right and 0 is not.
    
    (ra_new, dec_new) = wcs_in.wcs_pix2world(crpix[0] - dx_opnav, crpix[1] - dy_opnav, ORIGIN_FORMAT)

    # Set it
    
    wcs_in.wcs.crval = (ra_new, dec_new)
    
    return(wcs_in, (dy_opnav, dx_opnav))    

#==============================================================================
# Routine for testing only
#==============================================================================

def TESTING():

    import hbt
    import warnings
    import matplotlib.pyplot as plt

    dir =  '/Users/throop/Dropbox/Data/NH_Jring/data/jupiter/level2/lor/all/' 
    method_opnav = 'fft'

    plt.set_cmap('Greys_r')            
#    hbt.figsize((10,10))
    do_plot = True
    
    file_in = dir + 'lor_0034765323_0x630_sci_1.fit' # This one is faint -- hard to see much. But algo works -both.
#    file_in = dir + 'lor_0034602123_0x630_sci_1.fit'  # Algo works. Both g_t_i and fft.
#    file_in = dir + 'lor_0034613523_0x630_sci_1.fit'  # Fails fft. Works g_t_i.  
#    file_in = dir + 'lor_0034676528_0x630_sci_1.fit' # Good test case, very simple, good stars. g_t_i works great.
#    file_in = dir + 'lor_0034676528_0x630_sci_1_starnav.fit' # Post-navigation
#    file_in = dir + 'lor_0034676528_0x630_sci_1_starnav_starnav.fit' # Post-navigation
    file_in = dir + 'lor_0034604523_0x630_sci_1.fit' # shoudl be easy but it fails

    im = hbt.read_lorri(file_in) # Read the image, and process it a bit I think
    
    DO_PLOT = True
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")    
        w_orig  = WCS(file_in)           # Look up the WCS coordinates for this frame
        w       = WCS(file_in)           # Get a copy of it, which we'll change
    
    wcs_in = w  # Just for testing, leave this for cut+paste
    method = method_opnav # Leave for testing
    
    plt.set_cmap('Greys_r')
#    hbt.figsize((10,10))
    
    crval_orig = w_orig.wcs.crval
    
# Do the navigation call
    
    (w, (dy_pix, dx_pix)) = navigate_image_stellar(im, w, method = method_opnav,
                             title = file_in.split('/')[-1])
    
    crval = w.wcs.crval
    
# Now read the FITS header, so we can rewrite a revised version
        
    hdulist = fits.open(file) 
    header  = hdulist['PRIMARY'].header
    mode    = header['SFORMAT']     
    hdulist.close()  
    
    crval = w.wcs.crval

    print("Center was at RA {}, Dec {}".format(crval_orig[0], crval_orig[1]))
    print("Center is now at RA {}, Dec {}".format(crval[0], crval[1]))
    print("Determined OpNav offset: dx = {} pix, dy = {} pix".format(dx_pix, dy_pix))
    
# Read in the existing FITS file to memory, so we can modify it and write it back out
# Q: Is it just necessary to update these fields, or do I need to do more?

# A: Looks like just updating CRVAL1/2 in the text header itself should be enough.
#    See http://www.stsci.edu/hst/HST_overview/documents/multidrizzle/ch44.html

#    w.wcs.crval = (crval_new, x, y)

    header_wcs = w.to_header()   # NB: w.to_header, w.to_header(), and w.wcs.to_header() all do different things!
    
    hdulist = fits.open(file_in)
    header  = hdulist[0].header
    header['CRVAL1'] = crval[0] # RA, deg
    header['CRVAL2'] = crval[1] # Dec, deg
    header['OPNAVDX'] = (dx_pix, '[pix] Offset determined from stellar navigation, X')
    header['OPNAVDY'] = (dy_pix, '[pix] Offset determined from stellar navigation, Y')
    header['OPNAVMTH'] = (method_opnav, 'Method for OpNav stellar navigation')
    header['comment'] = 'OpNav CRVAL via navigate_image_stellar, ' + \
                         strftime("%Y-%m-%d %H:%M:%S UTC", gmtime())

# Write out a revised FITS file
    
    file_out = file_in.replace('.fit', '_starnav.fit')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")            # Ignore warning about a long comment.
        hdulist.writeto(file_out, overwrite=True)  # Write to a new file (open, write, and close, in one command)
        
    hdulist.close()                            # Close the original file
    
    print("Wrote: " + file_out)


                