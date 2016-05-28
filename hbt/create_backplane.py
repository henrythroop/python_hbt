# -*- coding: utf-8 -*-
"""
Created on Sat May 28 20:41:09 2016

@author: throop
"""

def create_backplane(image ):

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
from   scipy.optimize import curve_fit
#from   pylab import *  # So I can change plot size.
                       # Pylab defines the 'plot' command
import cspice
import skimage
from   itertools import izip    # To loop over groups in a table -- see astropy tables docs
from   astropy.wcs import WCS
from   astropy.vo.client import conesearch # Virtual Observatory, ie star catalogs
from   astropy import units as u           # Units library
from   astropy.coordinates import SkyCoord # To define coordinates to use in star search
#from   photutils import datasets
from   astropy.stats import sigma_clipped_stats
from   scipy.stats import mode
from   scipy.stats import linregress
from   photutils import daofind
import wcsaxes
import hbt

import imreg_dft as ird
import re # Regexp

# Time -- included
# Pointing - included
# Position -- included
# Input: Target name
# Possible backplanes made:
#  o Orbital distance (at equatorial plane)
#  o Azimuth angle (at equatorial plane)
#  o Phase angle, solar -- will vary only a tiny bit, but put in as an error check
#  o Sub-pt lon/lat -- no, wouldn't make any sense
#  o x y z position of this point (as error check)

# Create backplanes based on an image number. This is a stand-alone function, not part of the method

# For each pixel
# o Get its RA/Dec value from WCS
# o Compute a vector from s/c in that direction
# o Define a plane (from planet, perp to its rotational axis)
# o Use cspice.inrypl() to get intersection between ray and a plane. Probably returns J2K coords.
# o Convert J2K coords to Jupiter-system coords.
# o Convert those coords from rectangular to planetographic.
# o Now I have azimuth, radius, etc.
# o Compute phase angle from the rectangular coords

# This is probably the sort of things that MRS wants to avoid doing. I bet I can do 1000 coords and interpolate, and be
# nearly as accurate as computing 1 million points.

# Q: How do I deal with offsets?
#    o Modify the WCS pointing to be what is correct? Though slow if I am recomputing a lot.
#    o Shift the backplane by e.g., (-11, -18) pixels? Though will skip the edges.

    file = '/fits'
   
    file_tm = "/Users/throop/gv/dev/gv_kernels_new_horizons.txt"  # SPICE metakernel

    arr = get_image_nh(file)
    
    w = WCS(filename)
    
    header = hbt.get_image_header(filename)

    print 'crval[i] = ' + repr(w.wcs.crval)

    # Start up SPICE

    cspice.furnsh(file_tm)
# 
# Print some values. Not sure if this will work now that I have converted it to a string, not a dictionary...
    
    print 'ET = ' + header['ET']
    print 'EXPTIME =' + header['EXPTIME']

    et = header['ET']
    
# Create a plane based on Jupiter

    name_target = 'Jupiter'

    # Get the plane in Jupiter coordinates. Pretty easy!  Though we might want to do it in J2K coords

    plane_jup = cspice.nvp2pl([0,0,1], [0,0,0])    # nvp2pl: Normal Vec + Point to Plane


    
    # Q: How do you get the absolute position of a body... not relative to an observer, but relative to center of J2K frame?
    
    pos_jup = cspice.spkezr
    
       void nvp2pl_c ( ConstSpiceDouble    normal[3],
                   ConstSpiceDouble    point [3],
                   SpicePlane        * plane     )
                   
                   
    n_dx = 1024
    n_dy = 1024 # should get from NAXIS2
    
    for x in range(n_dx):
        for y in range(n_dy):
            ra, dec = w.wcs_pix2world(512, 512, 0) # Compute RA, Dec for a single pixel.
    
            vec_pix_j2k =  cspice.radrec(1., ra, dec) # Vector thru pixel, in J2K
            
            # Q: How can I transform this vector from J2K to jupiter system coords? Can I just run PXFORM on it? probably.
            
            mx_j2k_jup = cpsice.pxform()
# Get the vector from s/c, thru this pixel, thru the ring plane. Do it in Jupiter coordinates.
            
            # Find the intercept point between the ring plane and the vector, using cspice.inrypl()
            # Plane and vector must             
# How do I do that?        
            