# -*- coding: utf-8 -*-
"""
Created on Sat May 28 20:41:09 2016

@author: throop
"""


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

#def create_backplane():

file = '/Users/throop/Dropbox/Data/NH_Jring/data/jupiter/level2/lor/all/lor_0034609323_0x630_sci_1.fit'
   
file_tm = '/Users/throop/gv/dev/gv_kernels_new_horizons.txt'  # SPICE metakernel

#    arr = get_image_nh(file)

w = WCS(file)

header = hbt.get_image_header(file)

print 'crval[i] = ' + repr(w.wcs.crval)

arr = hbt.get_image_nh(file)

plt.imshow(arr)

# Start up SPICE

cspice.furnsh(file_tm)
# 
# Print some values. Not sure if this will work now that I have converted it to a string, not a dictionary...

hdulist = fits.open(file)

et = float(hdulist[0].header['SPCSCET']) # ET of mid-exposure, on s/c
exptime = float(hdulist[0].header['EXPOSURE']) / 1000 # Convert to seconds
utc = hdulist[0].header['SPCUTCID'] # UTC of mid-exposure, on s/c

print 'ET = ' + repr(et)
print 'EXPTIME =' + repr(exptime) + ' s'

hdulist.close()

# Create a plane based on Jupiter

name_target = 'Jupiter'

# Get the plane in Jupiter coordinates. Pretty easy!  Though we might want to do it in J2K coords

plane_jup = cspice.nvp2pl([0,0,1], [0,0,0])    # nvp2pl: Normal Vec + Point to Plane

# Q: How do you get the absolute position of a body... not relative to an observer, but relative to center of J2K frame?

pos_jup = cspice.spkezr

n_dx = 1024
n_dy = 1024 # should get from NAXIS2

lon_arr = np.zeros((n_dx, n_dy))
lat_arr = np.zeros((n_dx, n_dy))
radius_arr = np.zeros((n_dx, n_dy))
ra_arr = np.zeros((n_dx, n_dy))
dec_arr = np.zeros((n_dx, n_dy))

i_x = 512
i_y = 512

stop
# NB: at 2007-055T07:50:03.368, sub-obs lat = -6, d = 6.9 million km = 95 rj.
# sub-obs lon/lat = 350.2 / -6.7

rj = 71492  # Jupiter radius, in km. Just for q&d conversions

for i_x in range(n_dx):
    for i_y in range(n_dy):

        ra, dec = w.wcs_pix2world(i_x, i_y, 0) # Compute RA, Dec for a single pixel.
        
        ra = float(ra)          # cspice will crash if we give it a one-element array, instead of a scalar
        dec = float(dec)
        
#        print "ra,dec = " + repr(ra) + ", " + repr(dec)
#        print "et = " + repr(et)

        vec_pix_j2k =  cspice.radrec(1., ra*hbt.d2r, dec*hbt.d2r) # Vector thru pixel to ring, in J2K [args ok]
        
        # Q: How can I transform this vector from J2K to jupiter system coords? Can I just run PXFORM on it? probably.
        # A vector is basically the same as a point.
        
        mx_j2k_jup = cspice.pxform('J2000', 'IAU_JUPITER', et) # from, to, et

        # Convert vector along the pixel direction, from J2K into IAU_JUP frame
        # I *guess* I can do this... transform a vector from one frame into another.
  
        vec_pix_jup = cspice.mxv(mx_j2k_jup, vec_pix_j2k)
        
        # Get vec from Jup to NH, in Jupiter frame
               
        (st_jup_sc_jup, lt) = cspice.spkezr('New Horizons', et, 'IAU_JUPITER', 'LT', 'Jupiter')
        (st_jup_sc_j2k, lt) = cspice.spkezr('New Horizons', et, 'J2000',       'LT', 'Jupiter')
     
        vec_jup_sc_jup = st_jup_sc_jup[0:3]
        vec_jup_sc_j2k = st_jup_sc_j2k[0:3]
            
        # Name this vector a 'point'
        
        pt_jup_sc_jup = vec_jup_sc_jup
#        
#         void spkezr_c ( ConstSpiceChar     *targ,
#                   SpiceDouble         et,
#                   ConstSpiceChar     *ref,
#                   ConstSpiceChar     *abcorr,
#                   ConstSpiceChar     *obs,
#                   SpiceDouble         starg[6],
#                   SpiceDouble        *lt  
#        
        (npts, pt_intersect_jup) = cspice.inrypl(pt_jup_sc_jup, vec_pix_jup, plane_jup) # intersect ray and plane. Jup coords.

# Reality check: the vectors pt_jup_sc_jup and vec_pix_jup should both be very nearly opposite (within a degree or two)
# They're not -- they're off by 60 deg.
#
#        print 'angle = ' + repr(cspice.vsep(pt_jup_sc_jup, vec_pix_jup)*hbt.r2d)
        
# OK, for testing purposes, check that the angles are opposite each other when measured in J2K coords
# OK! These are also off by 60 deg. So we can figure out this problem in J2K.

#        print 'angle = ' + repr(cspice.vsep(vec_jup_sc_j2k, vec_pix_j2k)*hbt.r2d)
        
        radius, lon, lat = cspice.reclat(pt_intersect_jup)
        
        radius_arr[i_x, i_y] = radius
        lon_arr[i_x, i_y] = lon
        lat_arr[i_x, i_y] = lat
        ra_arr[i_x, i_y] = ra
        dec_arr[i_x, i_y] = dec
        
#        print 'Radius = ' + repr(radius)

#        print "ra,dec = {0},{1}; et = {2}; radius = {3}, lon/lat = {4}/{5}".format(ra, dec, et, radius, lon, lat)
#        print

        # For a reality check, compute the sub-observer angles. I think the GV values are computed with RECPGR,
        # which includes flattening and using a different E-W sense.
        
#        (radius, lon, lat) = cspice.reclat(vec_jup_sc_jup)
#   
#        print 'sub-obs lon/lat = ' + repr(lon*hbt.r2d) + '/' + repr(lat*hbt.r2d)
        
# Get the vector from s/c, thru this pixel, thru the ring plane. Do it in Jupiter coordinates.
        
        # Find the intercept point between the ring plane and the vector, using cspice.inrypl()
        # Plane and vector must             
# How do I do that?        
        