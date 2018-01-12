# -*- coding: utf-8 -*-
"""
Created on Sat May 28 20:41:09 2016

@author: throop
"""

import math      
import astropy
from   astropy.io import fits
import numpy as np
import spiceypy as sp
from   astropy.visualization import wcsaxes
import hbt
from   astropy.wcs import WCS
import os
import matplotlib.pyplot as plt

# Create backplanes based on an image number. This is a stand-alone function, not part of the method.
# It creates ones 

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

def create_backplane(file, 
                     frame         = 'IAU_JUPITER', 
                     name_target   = 'Jupiter', 
                     name_observer = 'New Horizons', 
                     type          = None):
    
    """
    Returns a set of backplanes for a single specified image. The image must have WCS coords available in its header.
    Backplanes include navigation info for every pixel, including RA, Dec, Eq Lon, Phase, etc.
    
    SPICE kernels must be alreaded loaded, and spiceypy running.
    
    Parameters
    ----
    
    file:
        String. Input filename, for FITS file.
    frame:
        String. Reference frame. 'IAU_JUPITER', 'IAU_MU69', '2014_MU69_SUNFLOWER_ROT', etc.
    name_target:
        String. Name of the central body. All geometry is referenced relative to this (e.g., radius, azimuth, etc)
    name_observer:
        String. Name of the observer. Must be a SPICE body name (e.g., 'New Horizons')
    type:
        String. Type of ring to assume for the backplane. Must be None, or 'Sunflower'.
        
    Output
    ----

    Output is a tuple, consisting of each of the backplanes. The size of each of these arrays is the same as the 
    input image.
    
    output = (ra, dec, radius_eq, longitude_eq, phase)
    
    Radius_eq:
        Radius, in the equatorial plane, in km
    Longitude_eq:
        Longitude, in the equatorial plane, in radians (0 .. 2pi)
    RA:
        RA of pixel, in radians
    Dec:
        Dec of pixel, in
    dRA:
        Projected offset in RA  direction between center of body (or barycenter) and pixel, in km.
    dDec:
        Projected offset in Dec direction between center of body (or barycenter) and pixel, in km.
        
    With special options selected (TBD), then additional backplanes will be generated -- e.g., a set of planes
    for each of the Jovian satellites in the image, or sunflower orbit, etc.
    
    No new FITS file is written. The only output is the returned data.
    
    """

    name_body = name_target  # Sometimes we use one, sometimes the other. Both are identical
    
    fov_lorri = 0.3 * hbt.d2r
   
    abcorr = 'LT+S'

    DO_SATELLITES = False  # Flag: Do we create an additional backplane for each of Jupiter's small sats?
    
    DO_TEST = False
    
    if DO_TEST:
        
        file = '/Users/throop/Dropbox/Data/NH_Jring/data/jupiter/level2/lor/all/lor_0034612923_0x630_sci_1.fit'       
        file_tm = '/Users/throop/gv/dev/gv_kernels_new_horizons.txt'  # SPICE metakernel
        sp.furnsh(file_tm)
        name_target = 'Jupiter'
        frame = 'IAU_JUPITER'
        name_observer = 'New Horizons'
    
    #    arr = get_image_nh(file)
#    file = '/Users/throop/Data/'
    
    w = WCS(file) # Warning: I have gotten a segfault here before if passing a FITS file with no WCS info.
    
    # Print some values. Not sure if this will work now that I have converted it to a string, not a dictionary...
    
    hdulist = fits.open(file)
    
    et      = float(hdulist[0].header['SPCSCET']) # ET of mid-exposure, on s/c
    
    n_dx    = int(hdulist[0].header['NAXIS1']) # Pixel dimensions of image. Both LORRI and MVIC have this.
    n_dy    = int(hdulist[0].header['NAXIS2'])
    
    hdulist.close()

    # Setup the output arrays
    
    lon_arr    = np.zeros((n_dy, n_dx))     # Longitude of pixel (defined with recpgr)
    lat_arr    = np.zeros((n_dy, n_dx))     # Latitude of pixel (which is zero, so meaningless)
    radius_arr = np.zeros((n_dy, n_dx))     # Radius, in km
    ra_arr     = np.zeros((n_dy, n_dx))     # RA of pixel
    dec_arr    = np.zeros((n_dy, n_dx))     # Dec of pixel
    dra_arr    = np.zeros((n_dy, n_dx))     # dRA of pixel: Distance in sky plane between pixel and body, in km. 
    ddec_arr   = np.zeros((n_dy, n_dx))     # dDec of pixel: Distande in sky plane between pixel and body, in km.
    phase_arr  = np.zeros((n_dy, n_dx))     # Phase angle    

# =============================================================================
#  Do the backplane, in the general case.
#  This is a long module, because we deal with all the satellites, the J-ring, etc.        
# =============================================================================
    
    if (True):
        
        # Look up body parameters, used for PGRREC().
    
        (num, radii) = sp.bodvrd(name_target, 'RADII', 3)
        
        r_e = radii[0]
        r_p = radii[2]
        flat = (r_e - r_p) / r_e

        # Define a SPICE 'plane' along the plane of the ring.
        # Do this in coordinate frame of the body (IAU_JUPITER, 2014_MU69_SUNFLOWER_ROT, etc).
                    
        if (name_target.upper() == 'JUPITER'):
            plane_target = sp.nvp2pl([0,0,1], [0,0,0])    # nvp2pl: Normal Vec + Point to Plane. Jupiter north pole?

            # For Jupiter only, define a few more output arrays for the final backplane set

            ang_metis_arr    = np.zeros((n_dy, n_dx))   # Angle from pixel to body, in radians
            ang_adrastea_arr = ang_metis_arr.copy()
            ang_thebe_arr    = ang_metis_arr.copy()
            ang_amalthea_arr = ang_metis_arr.copy()
        
        if ('MU69' in name_target.upper()):
            
        # Define a plane, which is the plane of sunflower rings (ie, X-Z plane in Sunflower frame)

            plane_target = sp.nvp2pl([0,1,0], [0,0,0]) # Normal Vec + Point to Plane. Use +Y (anti-sun) and origin

        # Get xformation matrix from J2K to jupiter system coords. I can use this for points *or* vectors.
                
        mx_j2k_frame = sp.pxform('J2000', frame, et) # from, to, et
        
        # Get vec from body to s/c, in both body frame, and J2K.
        # NB: The suffix _j2k indicates j2K frame. _frame indicates the frame of target (IAU_JUP, MU69_SUNFLOWER, etc)
               
        (st_target_sc_frame, lt) = sp.spkezr(name_observer, et, frame,   abcorr, name_target)
        (st_target_sc_j2k, lt)   = sp.spkezr(name_observer, et, 'J2000', abcorr, name_target)     
        
        vec_target_sc_frame = st_target_sc_frame[0:3]
        vec_target_sc_j2k   = st_target_sc_j2k[0:3]
        dist_target_sc      = sp.vnorm(vec_target_sc_j2k)   # Get target distance, in km
        
        # Name this vector a 'point'. INRYPL requires a point argument.
            
        pt_target_sc_frame = vec_target_sc_frame
    
        # Look up RA and Dec of target (from sc), in J2K 
        
        (_, ra_sc_target, dec_sc_target) = sp.recrad(-vec_target_sc_j2k)
        
        # Get vector from target to sun. We use this later for phase angle.
        
        (st_target_sun_frame, lt) = sp.spkezr('Sun', et, frame, abcorr, name_target) # From body to Sun, in body frame
        vec_target_sun_frame = st_target_sun_frame[0:3]
        
        # Create a 2D array of RA and Dec points
        
        xs = range(n_dx)
        ys = range(n_dy)
        (i_x_2d, i_y_2d) = np.meshgrid(xs, ys)
        (ra_2d, dec_2d) = w.wcs_pix2world(i_x_2d, i_y_2d, False) # Returns in degrees
        ra_2d  *= hbt.d2r                                        # Convert to radians
        dec_2d *= hbt.d2r
        
        # Compute the projected distance from MU69, in the sky plane, in km, for each pixel
        
        dra_arr = (ra_2d   - ra_sc_target)  * dist_target_sc / np.cos(dec_2d)
        ddec_arr = (dec_2d - dec_sc_target) * dist_target_sc  # Convert to km
    
    # Now compute position for additional bodies, as needed
    
        if (name_target.upper() == 'JUPITER'):
            vec_metis_j2k,lt     = sp.spkezr('Metis',    et, 'J2000', abcorr, 'New Horizons')
            vec_adrastea_j2k,lt  = sp.spkezr('Adrastea', et, 'J2000', abcorr, 'New Horizons')
            vec_thebe_j2k,lt     = sp.spkezr('Thebe',    et, 'J2000', abcorr, 'New Horizons')
            vec_amalthea_j2k,lt  = sp.spkezr('Amalthea', et, 'J2000', abcorr, 'New Horizons')
            
            vec_metis_j2k        = np.array(vec_metis_j2k[0:3])
            vec_thebe_j2k        = np.array(vec_thebe_j2k[0:3])
            vec_adrastea_j2k     = np.array(vec_adrastea_j2k[0:3])
            vec_amalthea_j2k     = np.array(vec_amalthea_j2k[0:3])
                
        for i_x in xs:
            for i_y in ys:
        
                # Look up the vector direction of this single pixel, which is defined by an RA and Dec
                # Vector is thru mpixel to ring, in J2K 
        
                vec_pix_j2k =  sp.radrec(1., ra_2d[i_y, i_x], dec_2d[i_y, i_x]) 
                
                # Convert vector along the pixel direction, from J2K into the target body frame
          
                vec_pix_frame = sp.mxv(mx_j2k_frame, vec_pix_j2k)
        
                # And calculate the intercept point between this vector, and the ring plane.
                # All these are in body coordinates.
                    
                (npts, pt_intersect_frame) = sp.inrypl(pt_target_sc_frame, vec_pix_frame, plane_target) # pt, vec, plane

                # In the case of MU69, the frame is defined s.t. the ring is in the XZ plane. This is strange, and 
                # I bet MU69 is the only ring like this. Swap it so that Z means 'vertical, out of plane.'
                
                if ('MU69' in name_target):
                    pt_intersect_frame = np.array([pt_intersect_frame[0], pt_intersect_frame[2], pt_intersect_frame[1]])
                
                # Get the radius ('alt') and azimuth ('lon') of the intersect, in the ring plane
                
#                lon, lat, alt = sp.recpgr(name_target, pt_intersect_frame, r_e, flat) # Returns (lon, lat, alt)
                radius_body, lon, lat = sp.reclat(pt_intersect_frame)                     # Returns (radius, lon, lat)

                # Calculate the phase angle: angle between s/c-to-ring, and ring-to-sun
        
                vec_ring_sun_frame = -pt_intersect_frame + vec_target_sun_frame
                
                angle_phase = sp.vsep(-vec_pix_frame, vec_ring_sun_frame)

                # Save various derived quantities
                         
                radius_arr[i_y, i_x] = radius_body  # RECPGR returns altitude, not radius. (RECLAT returns radius.)
                lon_arr[i_y, i_x]    = lon
                phase_arr[i_y, i_x]  = angle_phase
                
                # Now calc angular separation between this pixel, and the satellites in our list
                # Since these are huge arrays, cast into floats to make sure they are not doubles.
                
                if (name_body.upper() == 'JUPITER'):
                    ang_thebe_arr[i_y, i_x]    = sp.vsep(vec_pix_j2k, vec_thebe_j2k)
                    ang_adrastea_arr[i_y, i_x] = sp.vsep(vec_pix_j2k, vec_adrastea_j2k)
                    ang_metis_arr[i_y, i_x]    = sp.vsep(vec_pix_j2k, vec_metis_j2k)
                    ang_amalthea_arr[i_y, i_x] = sp.vsep(vec_pix_j2k, vec_amalthea_j2k) 

        # Assemble the results
    
        backplane = {
             'RA'           : ra_2d.astype(float),  # return radians
             'Dec'          : dec_2d.astype(float), # return radians 
             'dRA'          : dra_arr.astype(float),
             'dDec'         : ddec_arr.astype(float),
             'Radius_eq'    : radius_arr.astype(float),
             'Longitude_eq' : lon_arr.astype(float), 
             'Phase'        : phase_arr.astype(float)  }
        
        # In the case of Jupiter, add a few extra fields
        
        if (name_body.upper() == 'JUPITER'):
            backplane['Ang_Thebe']    = ang_thebe_arr.astype(float)   # Angle to Thebe, in radians
            backplane['Ang_Metis']    = ang_metis_arr.astype(float)
            backplane['Ang_Amalthea'] = ang_amalthea_arr.astype(float)
            backplane['Ang_Adrastea'] = ang_adrastea_arr.astype(float)
    
        # If distance to any of the small sats is < 0.3 deg, then delete that entry in the dictionary
        
            if (np.amin(ang_thebe_arr) > fov_lorri):
                del backplane['Ang_Thebe']
            else:
                print("Keeping Thebe".format(np.min(ang_thebe_arr) * hbt.r2d))
        
            if (np.amin(ang_metis_arr) > fov_lorri):
                del backplane['Ang_Metis']
            else:
                print("Keeping Metis, min = {} deg".format(np.min(ang_metis_arr) * hbt.r2d))
                
            if (np.amin(ang_amalthea_arr) > fov_lorri):
                del backplane['Ang_Amalthea']
            else:
                print("Keeping Amalthea, min = {} deg".format(np.amin(ang_amalthea_arr) * hbt.r2d))
        
            if (np.amin(ang_adrastea_arr) > fov_lorri):
                del backplane['Ang_Adrastea']
            else:
                print("Keeping Adrastea".format(np.min(ang_adrastea_arr) * hbt.r2d))
    
        
    # And return the backplane set
                 
    return backplane
    

# =============================================================================
#  Do the backplane for Jupiter
#  This is a long module, because we deal with all the satellites, the J-ring, etc.        
# =============================================================================
    
    if (name_target == 'Jupiter'):
        
        # Look up Jupiter body parameters, used for PGRREC().
    
        (num, radii) = sp.bodvrd(name_target, 'RADII', 3)
        
        r_e = radii[0]
        r_p = radii[2]
        flat = (r_e - r_p) / r_e

        # Define a spice 'plane' along Jupiter's equatorial plane.
            
        # Get the plane in Jupiter coordinates. Pretty easy!  Though we might want to do it in J2K coords
        
        plane_jup = sp.nvp2pl([0,0,1], [0,0,0])    # nvp2pl: Normal Vec + Point to Plane
        
        # Define a few more output arrays for the final backplane set
        
        ang_metis_arr    = np.zeros((n_dy, n_dx))   # Angle from pixel to body, in radians
        ang_adrastea_arr = ang_metis_arr.copy()
        ang_thebe_arr    = ang_metis_arr.copy()
        ang_amalthea_arr = ang_metis_arr.copy()
        
        # Get xformation matrix from J2K to jupiter system coords. I can use this for points *or* vectors.
                
        mx_j2k_jup = sp.pxform('J2000', frame, et) # from, to, et
        
        # Get vec from Jup to NH, in Jupiter frame (or whatever named frame is passed in)
                       
        (st_jup_sc_jup, lt) = sp.spkezr(name_observer, et, frame,         abcorr, name_target)
        (st_jup_sc_j2k, lt) = sp.spkezr(name_observer, et, 'J2000',       abcorr, name_target)     
        
        vec_jup_sc_jup = st_jup_sc_jup[0:3]
        vec_jup_sc_j2k = st_jup_sc_j2k[0:3]
    
        # Name this vector a 'point'
            
        pt_jup_sc_jup = vec_jup_sc_jup
    
        # Get vector from Jupiter to Sun
        # spkezr(target ... observer)
    
        (st_jup_sun_jup, lt) = sp.spkezr('Sun', et, frame, abcorr, name_target) # From Jup to Sun, in Jup frame
        vec_jup_sun_jup = st_jup_sun_jup[0:3]
        
        # NB: at 2007-055T07:50:03.368, sub-obs lat = -6, d = 6.9 million km = 95 rj.
        # sub-obs lon/lat = 350.2 / -6.7
        
    #    rj = 71492  # Jupiter radius, in km. Just for q&d conversions
        
        xs = range(n_dx)
        ys = range(n_dy)
        (i_x_2d, i_y_2d) = np.meshgrid(xs, ys)  # 5000 x 700: same shape as input MVIC image
        
        (ra_2d, dec_2d) = w.wcs_pix2world(i_x_2d, i_y_2d, False)
    
    # Now compute position for Adrastea, Metis, Thebe
    
        vec_metis_j2k,lt     = sp.spkezr('Metis',    et, 'J2000', abcorr, 'New Horizons')
        vec_adrastea_j2k,lt  = sp.spkezr('Adrastea', et, 'J2000', abcorr, 'New Horizons')
        vec_thebe_j2k,lt     = sp.spkezr('Thebe',    et, 'J2000', abcorr, 'New Horizons')
        vec_amalthea_j2k,lt  = sp.spkezr('Amalthea', et, 'J2000', abcorr, 'New Horizons')
        
        vec_metis_j2k        = np.array(vec_metis_j2k[0:3])
        vec_thebe_j2k        = np.array(vec_thebe_j2k[0:3])
        vec_adrastea_j2k     = np.array(vec_adrastea_j2k[0:3])
        vec_amalthea_j2k     = np.array(vec_amalthea_j2k[0:3])
        
    #    stop
        
        for i_x in xs:
            for i_y in ys:
        
                # Look up the vector direction of this single pixel, which is defined by an RA and Dec
                # Vector is thru mpixel to ring, in J2K 
        
                vec_pix_j2k =  sp.radrec(1., ra_2d[i_y, i_x]*hbt.d2r, dec_2d[i_y, i_x]*hbt.d2r) 
                
                # Convert vector along the pixel direction, from J2K into IAU_JUP frame
          
                vec_pix_jup = sp.mxv(mx_j2k_jup, vec_pix_j2k)
        
                # And calculate the intercept point between this vector, and the plane defined by Jupiter's equator.
                # Intersect ray and plane. Jup coords.
                    
                (npts, pt_intersect_jup) = sp.inrypl(pt_jup_sc_jup, vec_pix_jup, plane_jup) 
        
                # Now calculate the phase angle: angle between s/c-to-ring, and ring-to-sun
        
                vec_ring_sun_jup = -pt_intersect_jup + vec_jup_sun_jup
                
                angle_phase = sp.vsep(-vec_pix_jup, vec_ring_sun_jup)
    
                # Now calc angular separation between this pixel, and the satellites in our list
                # Since these are huge arrays, cast into floats to make sure they are not doubles.
                
                if DO_SATELLITES:
                    ang_thebe_arr[i_y, i_x] = sp.vsep(vec_pix_j2k, vec_thebe_j2k)
                    ang_adrastea_arr[i_y, i_x] = sp.vsep(vec_pix_j2k, vec_adrastea_j2k)
                    ang_metis_arr[i_y, i_x] = sp.vsep(vec_pix_j2k, vec_metis_j2k)
                    ang_amalthea_arr[i_y, i_x] = sp.vsep(vec_pix_j2k, vec_amalthea_j2k)
                    
                # Save various derived quantities
                
                lon, lat, alt = sp.recpgr(name_target, pt_intersect_jup, r_e, flat) # Returns (lon, lat, alt)
                alt, lon, lat = sp.reclat(pt_intersect_jup)                         # Returns (radius, lon, lat)
                
                radius_arr[i_y, i_x] = alt
                lon_arr[i_y, i_x]    = lon
                phase_arr[i_y, i_x]  = angle_phase
                
        # Assemble the results
    
        if DO_SATELLITES:
            backplane = {'RA'           : (ra_2d * hbt.d2r).astype(float), # return radians
                     'Dec'          : (dec_2d * hbt.d2r).astype(float), # return radians
                     'Radius_eq'    : radius_arr.astype(float),
                     'Longitude_eq' : lon_arr.astype(float), 
                     'Phase'        : phase_arr.astype(float),
                     'Ang_Thebe'    : ang_thebe_arr.astype(float),   # Angle to Thebe, in radians
                     'Ang_Metis'    : ang_metis_arr.astype(float),
                     'Ang_Amalthea' : ang_amalthea_arr.astype(float),
                     'Ang_Adrastea' : ang_adrastea_arr.astype(float)}
    
        # If distance to any of the small sats is < 0.3 deg, then delete that entry in the dictionary
        
            if (np.amin(ang_thebe_arr) > fov_lorri):
                del backplane['Ang_Thebe']
            else:
                print("Keeping Thebe".format(np.min(ang_thebe_arr) * hbt.r2d))
        
            if (np.amin(ang_metis_arr) > fov_lorri):
                del backplane['Ang_Metis']
            else:
                print("Keeping Metis, min = {} deg".format(np.min(ang_metis_arr) * hbt.r2d))
                
            if (np.amin(ang_amalthea_arr) > fov_lorri):
                del backplane['Ang_Amalthea']
            else:
                print("Keeping Amalthea, min = {} deg".format(np.amin(ang_amalthea_arr) * hbt.r2d))
        
            if (np.amin(ang_adrastea_arr) > fov_lorri):
                del backplane['Ang_Adrastea']
            else:
                print("Keeping Adrastea".format(np.min(ang_adrastea_arr) * hbt.r2d))
    
        else:
            backplane = {'RA'           : (ra_2d * hbt.d2r).astype(float), # return radians
                     'Dec'          : (dec_2d * hbt.d2r).astype(float), # return radians
                     'Radius_eq'    : radius_arr.astype(float),
                     'Longitude_eq' : lon_arr.astype(float), 
                     'Phase'        : phase_arr.astype(float)}

        
    # And return them
                 
    return backplane

##### END OF FUNCTION #####

# =============================================================================
# Do some tests to validate the function
# =============================================================================
    
if (__name__ == '__main__'):
    
    DO_TEST_JUPITER = False
    if (DO_TEST_JUPITER):
       
        file_in       = '/Users/throop/Dropbox/Data/NH_Jring/data/jupiter/level2/lor/all/lor_0034612923_0x630_sci_1.fit'
        file_tm       = '/Users/throop/gv/dev/gv_kernels_new_horizons.txt'  # SPICE metakernel
        sp.furnsh(file_tm)
        name_target   = 'Jupiter'
        frame         = 'IAU_JUPITER'
        name_observer = 'New Horizons'
   
    DO_TEST_MU69 = True
    if (DO_TEST_MU69):
        file_in = os.path.join(os.path.expanduser('~'), 'Data', 'NH_KEM_Hazard', 'ORT1_Jan18', 
                                   'pwcs_ort1','K1LR_MU69ApprField_115d_L2_2017264','lor_0368314467_0x633_pwcs.fits')
#                                   'lor_0406731132_0x633_sci_HAZARD_test1.fit')
    
        frame         = '2014_MU69_SUNFLOWER_ROT'
        name_target   = 'MU69'
        name_observer = 'New Horizons'
        file_tm = '/Users/throop/git/NH_rings/kernels_kem.tm'  # SPICE metakernel
        sp.furnsh(file_tm)

   
    if (DO_TEST_JUPITER or DO_TEST_MU69):
        
        planes = create_backplane(file_in, frame=frame, name_target=name_target, name_observer=name_observer)
    
    # If requested, plot all of the planes to the screen, for validation
    
        DO_PLOT = True
    
        if (DO_PLOT):
            i=1
            fig = plt.subplots()
            for key in planes.keys():
                plt.subplot(3,3,i)
                plt.imshow(planes[key])
                plt.title(key)
                i+=1
    
            plt.show()

    # Now write everything to a new FITS file. 
    
    file_out = file_in.replace('.fit', '_backplaned.fit')   # Works for both .fit and .fits
    
    # Open the existing file
    
    hdu = fits.open(file_in)
    
    # Go thru all of the new backplanes, and add them one by one. For each, create an ImageHDU, and then add it.
    
    for key in planes.keys():
        hdu_new = fits.ImageHDU(data=planes[key].astype(np.float32), name=key, header=None)
        hdu.append(hdu_new)
    
    # Write to a new file
    
    hdu.writeto(file_out, overwrite=True)
    print("Wrote: {}; {} planes; {:.1f} MB".format(file_out, 
                                                   len(hdu), 
                                                   os.path.getsize(file_out)/1e6))

    hdu.close()
    
#    hdu1 = fits.open(file_in)
#    hdu2 = fits.open(file_out)
#    
    # Now test the newly generated backplanes
    
    file_new = file_out
    stretch_percent = 90    
    stretch = astropy.visualization.PercentileInterval(stretch_percent)
    
    hdu = fits.open(file_new)
    
    # Start up SPICE
    
    file_kernel = '/Users/throop/git/NH_rings/kernels_kem.tm'
    sp.furnsh(file_kernel)
        
    # Look up position of MU69 in pixels.

    et = hdu[0].header['SPCSCET']
    utc = sp.et2utc(et, 'C', 0)
    abcorr = 'LT'
    frame = 'J2000'
    name_target = 'MU69'
    name_observer = 'New Horizons'
    w = WCS(file_new)
    
    (st,lt) = sp.spkezr(name_target, et, frame, abcorr, name_observer)
    vec_obs_mu69 = st[0:3]
    (_, ra, dec) = sp.recrad(vec_obs_mu69)
    (pos_pix_x, pos_pix_y) = w.wcs_world2pix(ra*hbt.r2d, dec*hbt.r2d, 0)
    
#    Plot the image itself
 
    hbt.figsize((10,10)) 
    plt.imshow(stretch(hdu[0].data))

    # Plot one of the planes

    plt.imshow(stretch(hdu['Longitude_eq'].data), alpha=0.5, cmap=plt.cm.Reds_r)

    # Plot the ring
 
    radius_ring = 100_000  # This needs to be adjusted for different distances.
    radius_arr = hdu['Radius_eq'].data
    radius_good = np.logical_and(radius_arr > radius_ring*0.95, radius_arr < radius_ring*1.05)
    plt.imshow(radius_good, alpha=0.3)
    
    # Plot MU69
    
    plt.plot(pos_pix_x, pos_pix_y, ms=10, marker = 'o', color='green')    
    plt.title("{}, {}".format(os.path.basename(file_new), utc))
    plt.show()    
