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
from   astropy.wcs import WCS

# Create backplanes based on an image number. This is a stand-alone function, not part of the method

# SPICE is required here, but it is *not* initialized. It is assumed that that has already been done.
# pds-tools does not implement the CSPICE.KTOTAL('ALL') function, which is the way I know of to list loaded 
# kernel files. So, just run something else first which initializes SPICE with a good kernel set.

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
#    o Shift the backplane by e.g., (-11, -18) pixels? This will crop some edges... but that is no big deal
#      since a lot of the edges are pretty contaminated by stray light and I can't really use them anyhow.
#
# I have some preference not to modify the original datafiles. 
# Also, I want to be able to tweak my centering position very slightly (few pixels), and get new radial profiles
# quickly. I guess this all means that I should make one set of backplanes based on the FITS WCS data, and
# shift them as neeeded.
#
# And, this means I should just make a script that does all the backplanes one time... rather than generating them 
# from within the GUI.

def create_backplane(file, frame = 'IAU_JUPITER', name_target='Jupiter', name_observer='New Horizons'):
    
    "Returns a set of backplanes for a given image. The image must have WCS coords available in its header."
    "Backplanes include navigation info for every pixel, including RA, Dec, Eq Lon, Phase, etc."

#    file = '/Users/throop/Dropbox/Data/NH_Jring/data/jupiter/level2/lor/all/lor_0034609323_0x630_sci_1.fit'
       
#    file_tm = '/Users/throop/gv/dev/gv_kernels_new_horizons.txt'  # SPICE metakernel
    
    #    arr = get_image_nh(file)
#    file = '/Users/throop/Data/NH_MVIC_Ring/mvic_d305_sum_mos_v1-new-image_fixed.fits'
    
    w = WCS(file) # Warning: I have gotten a segfault here before if passing a FITS file with no WCS info.
        
#    print 'crval[i] = ' + repr(w.wcs.crval)
    
#    image_arr = hbt.get_image_nh(file_image)
    
#    plt.imshow(image_arr)
    
    # Start up SPICE
    
#    cspice.furnsh(file_tm)

    # Print some values. Not sure if this will work now that I have converted it to a string, not a dictionary...
    
    hdulist = fits.open(file)
    
    et      = float(hdulist[0].header['SPCSCET']) # ET of mid-exposure, on s/c
#    exptime = float(hdulist[0].header['EXPOSURE']) / 1000 # Convert to seconds
#    utc     = hdulist[0].header['SPCUTCID'] # UTC of mid-exposure, on s/c
    
    n_dx    = int(hdulist[0].header['NAXIS1']) # Pixel dimensions of image. Both LORRI and MVIC have this.
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
    
    lon_arr    = np.zeros((n_dy, n_dx))     # Longitude of pixel (defined with recpgr)
    lat_arr    = np.zeros((n_dy, n_dx))     # Latitude of pixel (which is zero, so meaningless)
    radius_arr = np.zeros((n_dy, n_dx))     # Equatorial radius
    ra_arr     = np.zeros((n_dy, n_dx))     # RA of pixel
    dec_arr    = np.zeros((n_dy, n_dx))     # Dec of pixel
    phase_arr  = np.zeros((n_dy, n_dx))     # Phase angle    
    
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
    
    xs = range(n_dx)
    ys = range(n_dy)
    (i_x_2d, i_y_2d) = np.meshgrid(xs, ys)  # 5000 x 700: same shape as input MVIC image
    
    (ra_2d, dec_2d) = w.wcs_pix2world(i_x_2d, i_y_2d, False)
    
    for i_x in xs:
        for i_y in ys:
    
            # Look up the vector direction of this single pixel, which is defined by an RA and Dec
    
            vec_pix_j2k =  cspice.radrec(1., ra_2d[i_y, i_x]*hbt.d2r, dec_2d[i_y, i_x]*hbt.d2r) # Vector thru pixel to ring, in J2K [args ok]
            
            # Convert vector along the pixel direction, from J2K into IAU_JUP frame
      
            vec_pix_jup = cspice.mxv(mx_j2k_jup, vec_pix_j2k)
    
            # And calculate the intercept point between this vector, and the plane defined by Jupiter's equator
                
            (npts, pt_intersect_jup) = cspice.inrypl(pt_jup_sc_jup, vec_pix_jup, plane_jup) # intersect ray and plane. Jup coords.
    
            # Now calculate the phase angle: angle between s/c-to-ring, and ring-to-sun
    
            vec_ring_sun_jup = -pt_intersect_jup + vec_jup_sun_jup
            
            angle_phase = cspice.vsep(-vec_pix_jup, vec_ring_sun_jup)
            
            # Save various derived quantities
            
            lon, lat, alt = cspice.recpgr(name_target, pt_intersect_jup, r_e, flat)
            alt, lon, lat = cspice.reclat(pt_intersect_jup)
            
            radius_arr[i_y, i_x] = alt
            lon_arr[i_y, i_x] = lon
            phase_arr[i_y, i_x] = angle_phase
            
    # Assemble the results

    backplane = {'RA'           : ra_2d * hbt.d2r, # return radians
                 'Dec'          : dec_2d * hbt.d2r, # return radians
                 'Radius_eq'    : radius_arr,
                 'Longitude_eq' : lon_arr, 
                 'Phase'        : phase_arr}
    
    # And return them
                 
    return backplane

##### END OF FUNCTION #####


