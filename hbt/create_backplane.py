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


#from pylab import *
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

def create_backplane(file_image, frame = 'IAU_JUPITER', name_target='Jupiter', name_observer='New Horizons'):
    
    "Returns a set of backplanes for a given image. The image must have WCS coords available in its header."
    "Backplanes include navigation info for every pixel, including RA, Dec, Eq Lon, Phase, etc."

#    file = '/Users/throop/Dropbox/Data/NH_Jring/data/jupiter/level2/lor/all/lor_0034609323_0x630_sci_1.fit'
       
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
            alt, lon, lat = cspice.reclat(pt_intersect_jup)
            
            radius_arr[i_x, i_y] = alt
            lon_arr[i_x, i_y] = lon
            phase_arr[i_x, i_y] = angle_phase
            
    # Assemble the results

    backplane = {'RA'        : ra_2d * hbt.d2r, # return radians
                 'Dec'       : dec_2d * hbt.d2r, # return radians
                 'Radius_eq'    : radius_arr,
                 'Longitude_eq' : lon_arr, 
                 'Phase'        : phase_arr}
    
    # And return them
                 
    return backplane

##### END OF FUNCTION #####

#########
# Now write some code to run this function. This is just an example to create and plot the backplane for one image
##########
    
file = '/Users/throop/Dropbox/Data/NH_Jring/data/jupiter/level2/lor/all/lor_0034676944_0x630_sci_1.fit'
#file = '/Users/throop/Dropbox/Data/NH_Jring/data/jupiter/level2/lor/all/lor_0034676552_0x630_sci_1.fit'

plane = create_backplane(file)

arr_image = hbt.get_image_nh(file)

hbt.set_plot_defaults()
plt.rc('image', cmap='prism')               # Default color table for imshow
plt.rc('image', cmap='jet')               # Default color table for imshow

plt.rcParams['figure.figsize'] = 12, 18

# Make a 3 x 2 plot of all of the info, with a scalebar for each

fs = 20

fig = plt.figure()

ax1 = fig.add_subplot(3,2,1)
plot1 = plt.imshow(arr_image)
plt.title(file.split('/')[-1])
plt.xlim((0,1000))
plt.ylim((0,1000))
fig.colorbar(plot1)

ax2 = fig.add_subplot(3,2,2)
plot2 = plt.imshow(plane['Phase'] * hbt.r2d)
plt.title('Phase [deg]', fontsize=fs)
plt.xlim((0,1000))
plt.ylim((0,1000))
fig.colorbar(plot2)

ax3 = fig.add_subplot(3,2,3)
plot3 = plt.imshow(plane['RA'] * hbt.r2d)
plt.title('RA [deg]', fontsize=fs)
plt.xlim((0,1000))
plt.ylim((0,1000))
fig.colorbar(plot3)

ax4 = fig.add_subplot(3,2,4)
plot4 = plt.imshow(plane['Dec'] * hbt.r2d)
plt.title('Dec [deg]', fontsize=fs)
plt.xlim((0,1000))
plt.ylim((0,1000))
fig.colorbar(plot4)

ax5 = fig.add_subplot(3,2,5)
plot5 = plt.imshow(plane['Longitude_eq'] * hbt.r2d)
plt.title('Lon_eq [deg]', fontsize=fs)
plt.xlim((0,1000))
plt.ylim((0,1000))
fig.colorbar(plot5)

r_j = cspice.bodvrd('JUPITER', 'RADII')
ax6 = fig.add_subplot(3,2,6)
plot6 = plt.imshow(plane['Radius_eq'] / r_j[0])
plt.title('Radius_eq [$R_J$]', fontsize=fs)
plt.xlim((0,1000))
plt.ylim((0,1000))
fig.colorbar(plot6)

plt.show()

# J-ring is 122,000 .. 129,000 = 1.7 .. 1.8 R_J
radius = plane['Radius_eq'] / r_j[0]
azimuth = plane['Longitude_eq'] * hbt.r2d
#is_ring = np.array((radius > 122000)) & np.array((radius < 129000))
#plt.imshow(is_ring + arr_image / np.max(arr.image))

offset_dx = 59
offset_dy = 79
arr_image_roll = np.roll(np.roll(arr_image, offset_dx, axis=1), offset_dy, axis=0)

plt.imshow( ( np.array(radius > 1.7) & np.array(radius < 1.8)) + arr_image_roll / np.max(arr_image))
plt.show()

# Generate a radial profile

plt.rcParams['figure.figsize'] = 8,5

bin_radius = np.linspace(1.2, 2.1, num=200)
i_ring_r = np.zeros((200))
for i in range(np.size(bin_radius)-1):
    is_good = ( np.array(radius > bin_radius[i]) & np.array(radius < bin_radius[i+1]) )
    i_ring_r[i] = np.mean(arr_image_roll[is_good])

plt.plot(bin_radius, i_ring_r)  
plt.xlim((1.6,1.9))
plt.xlabel('$R_J$', fontsize=fs)
plt.ylabel('I', fontsize=fs)
plt.title('J-ring Radial Profile', fontsize=fs)
plt.show()
  
# Generate an azimuthal profile

bin_azimuth = np.linspace(-180,180, num=200)
i_ring_az = np.zeros((200))
for i in range(np.size(bin_azimuth)-1):
    is_good = ( np.array(radius > 1.7) & np.array(radius < 1.9) & 
                np.array(azimuth > bin_azimuth[i]) & np.array(azimuth < bin_azimuth[i+1]) )
    i_ring_az[i] = np.mean(arr_image_roll[is_good])
plt.plot(bin_azimuth, i_ring_az)
plt.xlim((50, 150))
plt.xlabel('Az [deg]', fontsize=fs)
plt.ylabel('I', fontsize=fs)
plt.title('J-ring Azimuthal Profile', fontsize=fs)
plt.show()   

i_az      = np.concatenate((i_ring_az[:-2], i_ring_az[:-2]))
az        = np.concatenate((bin_azimuth[:-2], bin_azimuth[:-2]+360))
plt.plot(az,i_az)
plt.xlim((0,540))
plt.show()

# Now, search this for the first pattern of [nan, non-nan]. Go from there, til next pattern of [non-nan, nan].

i = np.array(range(np.size(i_az)-1)) # just an index

is_start = np.isnan(i_az[i]) & np.logical_not(np.isnan(i_az[i+1]))
bin_start = i[is_start][0]+1 # get first match

# And then look for the end pattern: [non-nan, nan], starting *after* bin_start

is_end = np.logical_not(np.isnan(i_az[i])) & np.isnan(i_az[i+1]) & np.array(i > bin_start)
bin_end = i[is_end][0]

plt.plot(az,i_az)
plt.xlim((az[bin_start-2], az[bin_end+2]))
plt.show()


stop

