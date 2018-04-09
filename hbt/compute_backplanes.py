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

# Create backplanes based on an image number. This is a stand-alone function, *not* a class or method.

# SPICE is required here, but it is *not* initialized. It is assumed that that has already been done.

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

def compute_backplanes(file, name_target, frame, name_observer, angle1=0, angle2=0, angle3=0):
    
    """
    Returns a set of backplanes for a single specified image. The image must have WCS coords available in its header.
    Backplanes include navigation info for every pixel, including RA, Dec, Eq Lon, Phase, etc.
    
    The results are returned to memory, and not written to a file.
    
    SPICE kernels must be alreaded loaded, and spiceypy running.
    
    Parameters
    ----
    
    file:
        String. Input filename, for FITS file.
    frame:
        String. Reference frame of the target body. 'IAU_JUPITER', 'IAU_MU69', '2014_MU69_SUNFLOWER_ROT', etc.
                This is the frame that the Radius_eq and Longitude_eq are computed in.
    name_target:
        String. Name of the central body. All geometry is referenced relative to this (e.g., radius, azimuth, etc)
    name_observer:
        String. Name of the observer. Must be a SPICE body name (e.g., 'New Horizons')
    
    Optional Parameters
    ----
    
    angle{1,2,3}:
        Rotation angles which are applied when defining the plane in space that the backplane will be generated for.
        These are applied in the order 1, 2, 3. Angles are in radians. Nominal values are 0.
        
        This allows the simulation of (e.g.) a ring system inclined relative to the nominal body equatorial plane.
        
        For MU69 sunflower rings, the following descriptions are roughly accurate, becuase the +Y axis points
        sunward, which is *almost* toward the observer. But it is better to experiment and find the 
        appropriate angle that way, than rely on this ad hoc description. These are close for starting with.
 
                      `angle1` = Tilt front-back, from face-on. Or rotation angle, if tilted right-left.
                      `angle2` = Rotation angle, if tilted front-back. 
                      `angle3` = Tilt right-left, from face-on.
        
    Output
    ----

    Output is a tuple, consisting of each of the backplanes, and a text description for each one. 
    The size of each of these arrays is the same as the input image.
    
    The position of each of these is the plane defined by the target body, and the normal vector to the observer.
    
    output = (backplanes, descs)
    
        backplanes = (ra,      dec,      radius_eq,      longitude_eq,      phase)
        descs      = (desc_ra, desc_dec, desc_radius_eq, desc_longitude_eq, desc_phase)
    
    Radius_eq:
        Radius, in the ring's equatorial plane, in km
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
        
    Z:  
        Vertical value of the ring system.
        
    With special options selected (TBD), then additional backplanes will be generated -- e.g., a set of planes
    for each of the Jovian satellites in the image, or sunflower orbit, etc.
    
    No new FITS file is written. The only output is the returned tuple.
    
    """

    if not(frame):
        raise ValueError('frame undefined')
    
    if not(name_target):
        raise ValueError('name_target undefined')
        
    if not(name_observer):
        raise ValueError('name_observer undefined')
        
    name_body = name_target  # Sometimes we use one, sometimes the other. Both are identical
    
    fov_lorri = 0.3 * hbt.d2r
   
    abcorr = 'LT'

    do_satellites = False  # Flag: Do we create an additional backplane for each of Jupiter's small sats?
    
    # Open the FITS file
    
    w       = WCS(file) # Warning: I have gotten a segfault here before if passing a FITS file with no WCS info.    
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
    x_skyplane = np.zeros((n_dy, n_dx))     # Intersection of sky plane: X pos in bdoy coords
    y_skyplane = np.zeros((n_dy, n_dx))     # Intersection of sky plane: X pos in bdoy coords
    z_skyplane = np.zeros((n_dy, n_dx))     # Intersection of sky plane: X pos in bdoy coords
    

# =============================================================================
#  Do the backplane, in the general case.
#  This is a long routine, because we deal with all the satellites, the J-ring, etc.        
# =============================================================================
    
    if (True):
        
        # Look up body parameters, used for PGRREC().
    
        (num, radii) = sp.bodvrd(name_target, 'RADII', 3)
        
        r_e = radii[0]
        r_p = radii[2]
        flat = (r_e - r_p) / r_e

        # Define a SPICE 'plane' along the plane of the ring.
        # Do this in coordinate frame of the body (IAU_JUPITER, 2014_MU69_SUNFLOWER_ROT, etc).
        
# =============================================================================
# Set up the Jupiter system specifics
# =============================================================================
            
        if (name_target.upper() == 'JUPITER'):
            plane_target_eq = sp.nvp2pl([0,0,1], [0,0,0])    # nvp2pl: Normal Vec + Point to Plane. Jupiter north pole?

            # For Jupiter only, define a few more output arrays for the final backplane set

            ang_metis_arr    = np.zeros((n_dy, n_dx))   # Angle from pixel to body, in radians
            ang_adrastea_arr = ang_metis_arr.copy()
            ang_thebe_arr    = ang_metis_arr.copy()
            ang_amalthea_arr = ang_metis_arr.copy()

# =============================================================================
# Set up the MU69 specifics
# =============================================================================
        
        if ('MU69' in name_target.upper()):
            
        # Define a plane, which is the plane of sunflower rings (ie, X-Z plane in Sunflower frame)
        # If additional angles are passed, then create an Euler matrix which will do additional angles of rotation.
        # This is defined in the 'MU69_SUNFLOWER' frame

            mx_euler = sp.eul2m(angle3, angle2, angle1, 3, 2, 1)  # (1, 2, 3) refers to axes (x, y, z)
            vec_plane = [0, 1, 0]                                 # Use +Y (anti-sun dir)
            vec_plane_tilted = sp.mxv(mx_euler, vec_plane)
            plane_target_eq = sp.nvp2pl(vec_plane_tilted, [0,0,0]) # "Normal Vec + Point to Plane". 0,0,0 = origin.

# =============================================================================
# Set up the various output planes and arrays necessary for computation
# =============================================================================

        # Get xformation matrix from J2K to target system coords. I can use this for points *or* vectors.
                
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
        
        # Set up a plane normal to the observer, that goes through the target. 
        # It should be centered on the target. It basically defines the 'sky plane.'
        # By convention, set up vector so it points from body, out into anti-observer direction.
        # Set this up in J2K.
        
        pt_plane_obs_targ_norm_j2k = sp.spkezr('Sun', et, 'J2000', abcorr, name_target)[0][0:3]  # Find abs pos of body
        
        vec_plane_obs_targ_norm_j2k = -vec_target_sc_j2k                                        # Find normal vec
        
        plane_target_sky = sp.nvp2pl(vec_plane_obs_targ_norm_j2k, pt_plane_obs_targ_norm_j2k) # Create the plane
        
        # Compute the position of the observer, relative to Sun, in J2K coords. This is for skyplane calc below.
        # Also get position of target, relative to Sun.
        # I could equally well do these as position wrt SS barycenter, but I don't know body name for that.
        
        pt_obs_j2k    = sp.spkezr('Sun', et, 'J2000', abcorr, name_observer)[0][0:3]
        pt_target_j2k = sp.spkezr('Sun', et, 'J2000', abcorr, name_observer)[0][0:3]
        
        # Create a 2D array of RA and Dec points
        
        xs = range(n_dx)
        ys = range(n_dy)
        (i_x_2d, i_y_2d) = np.meshgrid(xs, ys)
        (ra_arr, dec_arr) = w.wcs_pix2world(i_x_2d, i_y_2d, False) # Returns in degrees
        ra_arr  *= hbt.d2r                                        # Convert to radians
        dec_arr *= hbt.d2r
        
        # Compute the projected distance from MU69, in the sky plane, in km, for each pixel
        
        dra_arr  = (ra_arr   - ra_sc_target) * dist_target_sc / np.cos(dec_arr)
        ddec_arr = (dec_arr - dec_sc_target) * dist_target_sc  # Convert to km
    
# =============================================================================
#  Compute position for additional Jupiter bodies, as needed
# =============================================================================
    
        if (name_target.upper() == 'JUPITER'):
            vec_metis_j2k,lt     = sp.spkezr('Metis',    et, 'J2000', abcorr, 'New Horizons')
            vec_adrastea_j2k,lt  = sp.spkezr('Adrastea', et, 'J2000', abcorr, 'New Horizons')
            vec_thebe_j2k,lt     = sp.spkezr('Thebe',    et, 'J2000', abcorr, 'New Horizons')
            vec_amalthea_j2k,lt  = sp.spkezr('Amalthea', et, 'J2000', abcorr, 'New Horizons')
            
            vec_metis_j2k        = np.array(vec_metis_j2k[0:3])
            vec_thebe_j2k        = np.array(vec_thebe_j2k[0:3])
            vec_adrastea_j2k     = np.array(vec_adrastea_j2k[0:3])
            vec_amalthea_j2k     = np.array(vec_amalthea_j2k[0:3])
        
# =============================================================================
# Loop over pixels in the output image
# =============================================================================
        
        for i_x in xs:
            for i_y in ys:
        
                # Look up the vector direction of this single pixel, which is defined by an RA and Dec
                # Vector is thru pixel to ring, in J2K. 
        
                vec_pix_j2k =  sp.radrec(1., ra_arr[i_y, i_x], dec_arr[i_y, i_x]) 
                
                # Convert vector along the pixel direction, from J2K into the target body frame
          
                vec_pix_frame = sp.mxv(mx_j2k_frame, vec_pix_j2k)
        
                # And calculate the intercept point between this vector, and the ring plane.
                # All these are in body coordinates.
                    
                (npts, pt_intersect_frame) = sp.inrypl(pt_target_sc_frame, vec_pix_frame, plane_target_eq) 
                                                                                             # pt, vec, plane

                # In the case of MU69, the frame is defined s.t. the ring is in the XZ plane. This is strange, and 
                # I bet MU69 is the only ring like this. Swap it so that Z means 'vertical, out of plane.'
                
                if ('MU69' in name_target):
                    pt_intersect_frame = np.array([pt_intersect_frame[0], pt_intersect_frame[2], pt_intersect_frame[1]])
                
                # Get the radius and azimuth of the intersect, in the ring plane
                
                radius_body, lon, lat = sp.reclat(pt_intersect_frame)

                # Calculate the phase angle: angle between s/c-to-ring, and ring-to-sun
        
                vec_ring_sun_frame = -pt_intersect_frame + vec_target_sun_frame
                
                angle_phase = sp.vsep(-vec_pix_frame, vec_ring_sun_frame)

                # Calc the vertical position ('Z') in the sky plane. This is useful for edge-on rings.
                # To do this, take the vector for this pixel, and find intersection 
                # with skyplane.
                # Arguments: INRYPL(pt_of_vec, dir_of_vec, plane_to_intersect
                #     Pt = sun-to-observer
                #     vector = observer-to-pixel ray
                #     plane = (defined sky plane variable)
                
                # Now get the intersection point. It is returned in the same frame as the ray, which is J2K
                (npts, pt_intersect_j2k) = sp.inrypl(pt_obs_j2k, vec_pix_j2k, plane_target_sky)
                
                # Now take this point, and convert into a vertical position in the body frame.
                # To do this transform the pt from J2K coords, into body frame, and then take z
                
                pt_intersect_j2k_relative = pt_intersect_j2k - pt_target_j2k
                vec_pix_frame = sp.mxv(mx_j2k_frame, pt_intersect_j2k_relative)  # Now this is XYZ in MU69 coords

                if ('MU69' in name_target):
                    pt_intersect_skyplane = pt_intersect_frame # Q: Do we need to swap XYZ here? Not sure.
                    
#                                     = np.array([pt_intersect_frame[0], pt_intersect_frame[2], pt_intersect_frame[1]])

                # Save various derived quantities
                         
                radius_arr[i_y, i_x] = radius_body  # RECPGR returns altitude, not radius. (RECLAT returns radius.)
                lon_arr[i_y, i_x]    = lon
                phase_arr[i_y, i_x]  = angle_phase
                x_skyplane[i_y, i_x] = pt_intersect_frame[0]
                y_skyplane[i_y, i_x] = pt_intersect_frame[1]
                z_skyplane[i_y, i_x] = pt_intersect_frame[2]
                
                
                # Now calc angular separation between this pixel, and the satellites in our list
                # Since these are huge arrays, cast into floats to make sure they are not doubles.
                
                if (name_body.upper() == 'JUPITER'):
                    ang_thebe_arr[i_y, i_x]    = sp.vsep(vec_pix_j2k, vec_thebe_j2k)
                    ang_adrastea_arr[i_y, i_x] = sp.vsep(vec_pix_j2k, vec_adrastea_j2k)
                    ang_metis_arr[i_y, i_x]    = sp.vsep(vec_pix_j2k, vec_metis_j2k)
                    ang_amalthea_arr[i_y, i_x] = sp.vsep(vec_pix_j2k, vec_amalthea_j2k) 

        # Assemble the results into a backplane
    
        backplane = {
             'RA'           : ra_arr.astype(float),  # return radians
             'Dec'          : dec_arr.astype(float), # return radians 
             'dRA_km'       : dra_arr.astype(float),
             'dDec_km'      : ddec_arr.astype(float),
             'Radius_eq'    : radius_arr.astype(float),
             'Longitude_eq' : lon_arr.astype(float), 
             'Phase'        : phase_arr.astype(float),
             'X_sky'        : x_skyplane.astype(float),
             'Y_sky'        : y_skyplane.astype(float),  # Looks like 'Y' is the one I want for vertical position
             'Z_sky'        : z_skyplane.astype(float)
             }
        
        # Assemble a bunch of descriptors, to be put into the FITS headers
        
        desc = {
                'RA of pixel, radians',
                'Dec of pixel, radians',
                'Offset from target in target plane, RA direction, km',
                'Offset from target in target plane, Dec direction, km',
                'Projected equatorial radius, km',
                'Projected equatorial longitude, km',
                'Sun-target-observer phase angle, radians',
                'X position of sky plane intercept',
                'Y position of sky plane intercept',
                'Z position of sky plane intercept'
                }
                
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
                 
    return (backplane, desc)

# =============================================================================
# End of function
# =============================================================================

# =============================================================================
# Do some tests to validate the function
# This creates files in memory, but does not write to file    
# =============================================================================
    
if (__name__ == '__main__'):
    
    import  matplotlib.pyplot as plt
    
    do_test_jupiter = False
    do_test_mu69    = True
    do_plot         = True
    
    if (do_test_jupiter):
       
        file_in       = '/Users/throop/Dropbox/Data/NH_Jring/data/jupiter/level2/lor/all/lor_0034612923_0x630_sci_1.fit'
        file_tm       = '/Users/throop/gv/dev/gv_kernels_new_horizons.txt'  # SPICE metakernel
        name_target   = 'Jupiter'
        frame         = 'IAU_JUPITER'
        name_observer = 'New Horizons'
       
    if (do_test_mu69):

        file_in       = '/Users/throop/Data/ORT1/porter/pwcs_ort1/K1LR_HAZ00/lor_0405175932_0x633_pwcs.fits'
        frame         = '2014_MU69_SUNFLOWER_ROT'
        name_target   = 'MU69'
        name_observer = 'New Horizons'
        file_tm       = '/Users/throop/git/NH_rings/kernels_kem_prime.tm'  # SPICE metakernel
   
    if (do_test_jupiter or do_test_mu69):

         # Start SPICE, if necessary
    
        if (sp.ktotal('ALL') == 0):
            sp.furnsh(file_tm)
               
        # Create the backplanes in memory
        
        (planes, desc) = compute_backplanes(file_in, name_target, frame, name_observer,
                      angle1=88*hbt.d2r,  # Tilt front-back, from face-on. Or rotation angle, if tilted right-left.
                      angle2=00*hbt.d2r,  # Or rotation angle, if tilted front-back. 
                      angle3=30*hbt.d2r)  # Tilt right-left, from face-on

        print("Backplanes generated for {}".format(file_in))
        
# =============================================================================
# Plot the newly generated backplanes, if requested
# =============================================================================
            
        if do_plot:
            
            hbt.fontsize(8)
            nxy = math.ceil(math.sqrt(len(planes)))  # Compute the grid size needed to plot all the planes to screen
            
            # Loop and plot each subplane individually, as color gradient plots.
            
            i = 1
            fig = plt.subplots()
            for key in planes.keys():
                plt.subplot(nxy,nxy,i)
                plt.imshow(planes[key])
                plt.title(key)
                plt.colorbar()
                i+=1

            # Make a few more plots as contour plots. Easier to see this way.
            
            planes_extra = ['Radius_eq', 'Longitude_eq']

            for key in planes_extra:
                
                plt.subplot(nxy, nxy, i)
                plt.contour(planes[key], aspect=1)
                plt.ylim((hbt.sizex(planes[key]), 0))
                plt.gca().set_aspect('equal')
                plt.title(key)
                i+=1
            
            plt.tight_layout()
            plt.show()