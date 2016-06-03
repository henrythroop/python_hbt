# -*- coding: utf-8 -*-
"""
Created on Sat May 28 20:41:09 2016

@author: throop
"""

import math      
import astropy
from   astropy.io import fits
import numpy as np
import cspice
import wcsaxes
import hbt

from pylab import *
# Create backplanes based on an image number. This is a stand-alone function, not part of the method

# For each pixel, do the followiung:
# o Get its RA/Dec value from WCS
# o Compute a vector from s/c in that direction
# o Define a plane (from planet, perp to its rotational axis)
# o Use cspice.inrypl() to get intersection between ray and a plane (in Jupiter coords).
# o Convert those coords from rectangular to planetographic.
# o Now I have azimuth, radius, etc.
# o Compute phase angle from the rectangular coords
#   *** For LORRI, the variation in phase angle across frame is pretty small. 

# Q: How do I deal with offsets?
#    o Modify the WCS pointing to be what is correct? Though slow if I am recomputing a lot.
#    o Shift the backplane by e.g., (-11, -18) pixels? Though will skip the edges.

# 
def create_backplane(file_image, frame = 'IAU_JUPITER', name_target='Jupiter', name_observer='New Horizons'):
    
    "Returns a set of backplanes for a given image. The image must have WCS coords available in its header."
    "Backplanes include navigation info for every pixel, including RA, Dec, Eq Lon, Phase, etc."

    file = '/Users/throop/Dropbox/Data/NH_Jring/data/jupiter/level2/lor/all/lor_0034609323_0x630_sci_1.fit'
       
    file_tm = '/Users/throop/gv/dev/gv_kernels_new_horizons.txt'  # SPICE metakernel
    
    #    arr = get_image_nh(file)
    
    w = WCS(file)
    
    header = hbt.get_image_header(file_image)
    
#    print 'crval[i] = ' + repr(w.wcs.crval)
    
#    image_arr = hbt.get_image_nh(file_image)
    
#    plt.imshow(image_arr)
    
    # Start up SPICE
    
#    cspice.furnsh(file_tm)

    # Print some values. Not sure if this will work now that I have converted it to a string, not a dictionary...
    
    hdulist = fits.open(file)
    
    et      = float(hdulist[0].header['SPCSCET']) # ET of mid-exposure, on s/c
    exptime = float(hdulist[0].header['EXPOSURE']) / 1000 # Convert to seconds
    utc     = hdulist[0].header['SPCUTCID'] # UTC of mid-exposure, on s/c
    
    n_dx    = int(hdulist[0].header['NAXIS1']) # Pixel dimensions of image
    n_dy    = int(hdulist[0].header['NAXIS2'])
    
#    print 'ET = ' + repr(et)
#    print 'EXPTIME =' + repr(exptime) + ' s'
    
    hdulist.close()
    
    # Look up parameters for the target body, used for PGRREC().
    
    radii = cspice.bodvrd(name_target, 'RADII')
    
    r_e = radii[0]
    r_p = radii[2]
    flat = (r_e - r_p) / r_e
    
    # Define a spice 'plane' along Jupiter's equatorial plane.
        
    # Get the plane in Jupiter coordinates. Pretty easy!  Though we might want to do it in J2K coords
    
    plane_jup = cspice.nvp2pl([0,0,1], [0,0,0])    # nvp2pl: Normal Vec + Point to Plane
    
    lon_arr = np.zeros((n_dx, n_dy))        # Longitude of pixel (defined with recpgr)
    lat_arr = np.zeros((n_dx, n_dy))        # Latitude of pixel (which is zero, so meaningless)
    radius_arr = np.zeros((n_dx, n_dy))     # Equatorial radius
    ra_arr = np.zeros((n_dx, n_dy))         # RA of pixel
    dec_arr = np.zeros((n_dx, n_dy))        # Dec of pixel
    phase_arr = np.zeros((n_dx, n_dy))      # Phase angle    
    
    # Get xformation matrix from J2K to jupiter system coords. I can use this for points *or* vectors.
            
    mx_j2k_jup = cspice.pxform('J2000', frame, et) # from, to, et
    
    # Get vec from Jup to NH, in Jupiter frame (or whatever named frame is passed in)
                   
    (st_jup_sc_jup, lt) = cspice.spkezr(name_observer, et, frame,         'LT', name_target)
    (st_jup_sc_j2k, lt) = cspice.spkezr(name_observer, et, 'J2000',       'LT', name_target)     
    
    vec_jup_sc_jup = st_jup_sc_jup[0:3]
    vec_jup_sc_j2k = st_jup_sc_j2k[0:3]

    # Name this vector a 'point'
        
    pt_jup_sc_jup = vec_jup_sc_jup

    # Get vector from Jupiter to Sun
    # spkezr(target ... observer)

    (st_jup_sun_jup, lt) = cspice.spkezr('Sun', et, frame, 'LT', name_target) # From Jup to Sun, in Jup frame
    vec_jup_sun_jup = st_jup_sun_jup[0:3]
    
    # NB: at 2007-055T07:50:03.368, sub-obs lat = -6, d = 6.9 million km = 95 rj.
    # sub-obs lon/lat = 350.2 / -6.7
    
#    rj = 71492  # Jupiter radius, in km. Just for q&d conversions
    
    i_y_2d = np.outer(range(n_dy), 1 + np.zeros(n_dy))
    i_x_2d = np.transpose(i_y_2d)
    (ra_2d, dec_2d) = w.wcs_pix2world(i_x_2d, i_y_2d, False)
    
    for i_x in range(n_dx):
        for i_y in range(n_dy):
    
            # Look up the vector direction of this single pixel, which is defined by an RA and Dec
    
            vec_pix_j2k =  cspice.radrec(1., ra_2d[i_x, i_y]*hbt.d2r, dec_2d[i_x, i_y]*hbt.d2r) # Vector thru pixel to ring, in J2K [args ok]
            
            # Convert vector along the pixel direction, from J2K into IAU_JUP frame
      
            vec_pix_jup = cspice.mxv(mx_j2k_jup, vec_pix_j2k)
    
            # And calculate the intercept point between this vector, and the plane defined by Jupiter's equator
                
            (npts, pt_intersect_jup) = cspice.inrypl(pt_jup_sc_jup, vec_pix_jup, plane_jup) # intersect ray and plane. Jup coords.
    
            # Now calculate the phase angle: angle between s/c-to-ring, and ring-to-sun
    
            vec_ring_sun_jup = -pt_intersect_jup + vec_jup_sun_jup
            
            angle_phase = cspice.vsep(-vec_pix_jup, vec_ring_sun_jup)
            
            # Save various derived quantities
            
            lon, lat, alt = cspice.recpgr(name_target, pt_intersect_jup, r_e, flat)
            
            radius_arr[i_x, i_y] = alt
            lon_arr[i_x, i_y] = lon
            phase_arr[i_x, i_y] = angle_phase
            
    # Assemble the results

    backplane = {'RA'        : ra_2d,
                 'Dec'       : dec_2d,
                 'Radius_eq'    : radius_arr,
                 'Longitude_eq' : lon_arr, 
                 'Phase'        : phase_arr}
    
    # And return them
                 
    return backplane

# Now write some code to run this function. I think I can do this here?
    
file = '/Users/throop/Dropbox/Data/NH_Jring/data/jupiter/level2/lor/all/lor_0034609323_0x630_sci_1.fit'

plane = create_backplane(file)

arr_image = hbt.get_image_nh(file)

hbt.set_plot_defaults()
plt.rc('image', cmap='prism')               # Default color table for imshow
plt.rc('image', cmap='jet')               # Default color table for imshow

rcParams['figure.figsize'] = 12, 18

# Make a 3 x 2 plot of all of the info, with a scalebar for each

fs = 20

plt.subplot(3,2,1)
plt.imshow(arr_image)
plt.title(file)

plt.subplot(3,2,2)
plt.imshow(plane['Phase'])
plt.title('Phase', fontsize=fs)

plt.subplot(3,2,3)
plt.imshow(plane['RA'])
plt.title('RA', fontsize=fs)

plt.subplot(3,2,4)
plt.imshow(plane['Dec'])
plt.title('Dec', fontsize=fs)

plt.subplot(3,2,5)
plt.imshow(plane['Longitude_eq'])
plt.title('Lon_eq', fontsize=fs)

plt.subplot(3,2,6)
plt.imshow(plane['Radius_eq'])
plt.title('Radius_eq', fontsize=fs)

plt.show()

radius = plane['Radius_eq']
is_ring = np.array((radius > 122000)) & np.array((radius < 129000))
plt.imshow(is_ring + arr_image / np.max(arr.image))

plt.imshow( ( np.array(radius > 61000) & np.array(radius < 64500)) + arr_image / np.max(arr_image))
stop

