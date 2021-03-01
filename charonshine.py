#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 23:38:10 2021

@author: hthroop
"""
# HBT 22-Jan-2021
# 
# Short program to calculate Charonshin on Pluto.
# Written for Tod Lauer, for his paper.

import spiceypy as sp

import numpy as np

import os
import matplotlib.pyplot as plt

dir_sp = "/Volumes/SSD External"
dir_sp = "/Users/hthroop/kernels/"
file_tm = 'gv_kernels_new_horizons.txt'

os.chdir(dir_sp)

import math

r2d = 360 / (2 * math.pi)
d2r = 1/r2d

sp.furnsh(os.path.join(dir_sp, file_tm))

ut =  "2015 July 15 01:58:00"
et = sp.utc2et(ut)

num_lat = 191 # Number of latitude points. Longitude point count is double this. 
              # 100 is good for rough; 300 is good for final plot

num_lat = num_lat
num_lon = num_lat * 2

lat = np.linspace(-90,  90, num=num_lat)
lon = np.linspace(0,   360, num=num_lon)

lon_rad = lon * d2r
lat_rad = lat * d2r

# Make a 2D array, from 2 1D arrays. This is my previous comult() function.
# Note that indexing = 'xy' is the default, but I do not want that here (it is swapped).

(lon_2d, lat_2d) = np.meshgrid(lon,lat, indexing = 'ij') # This is the one to use.

ang_pluto_sun    = np.zeros([num_lon, num_lat])  # Angle from Pluto surface point to Sun, rad
ang_pluto_charon = np.zeros([num_lon, num_lat])  # Angle from Pluto surface point to Charon, rad
ang_pluto_sc     = np.zeros([num_lon, num_lat])  # Angle from Pluto surface point to Sun, rad

ang_pluto_sun_2    = np.zeros([num_lon, num_lat])  # Angle from Pluto surface point to Sun, rad
ang_pluto_charon_2 = np.zeros([num_lon, num_lat])  # Angle from Pluto surface point to Charon, rad
ang_pluto_sc_2     = np.zeros([num_lon, num_lat])  # Angle from Pluto surface point to Sun, rad

frame = 'J2000'
abcorr = 'LT'

radius_pluto = 1188 # km
radius_charon = 606 # km

(state_pluto_charon, _) = sp.spkezr('Charon',       et, frame, abcorr, 'Pluto') # From P to C
(state_pluto_sc,     _) = sp.spkezr('New Horizons', et, frame, abcorr, 'Pluto') # From P to SC
(state_pluto_sun,    _) = sp.spkezr('Sun',          et, frame, abcorr, 'Pluto') # From P to Sun

vec_pluto_charon = state_pluto_charon[0:3]
vec_pluto_sc     = state_pluto_sc[0:3]
vec_pluto_sun    = state_pluto_sun[0:3]

mx = sp.pxform('IAU_PLUTO', frame, et)

for i,lat_i in enumerate(lat_rad):
    for j,lon_j in enumerate(lon_rad):
        
        vec_pluto_psurf_iau   = sp.latrec(radius_pluto, lon_j, lat_i) # From P_center to P_surface
        vec_pluto_psurf = np.dot(mx, vec_pluto_psurf_iau) # COnvert to J2000
        
        # Pluto-Sun angle, for this lon-lat location
        
        vec_psurf_sun = vec_pluto_sun - vec_pluto_psurf
        ang = sp.vsep(vec_psurf_sun, vec_pluto_psurf)
        ang = sp.vsep(vec_pluto_sun, vec_pluto_psurf)
        ang_pluto_sun[j,i] = ang
        
        # Pluto-Charon angle
        
        vec_psurf_charon = vec_pluto_charon - vec_pluto_psurf
        ang = sp.vsep(vec_psurf_charon, vec_pluto_psurf)
        ang_pluto_charon[j,i] = ang
        
        # Pluto-SC angle
        
        vec_psurf_sc = vec_pluto_sc - vec_pluto_psurf
        ang = sp.vsep(vec_psurf_sc, vec_pluto_psurf)
        ang_pluto_sc[j,i] = ang
    
        # Now, do this all a second way, just to validate.
        
        (trgepc, srfvec, phase, solar, emmisn) = sp.ilumin('ELLIPSOID', 'PLUTO', et, 'IAU_PLUTO', abcorr, 'Charon', vec_pluto_psurf_iau)
        
        ang_pluto_sun_2[j,i] = solar      # Solar incidence angle, from normal, at point. THIS MATCHES -- GOOD!
        ang_pluto_charon_2[j,i] = emmisn  # Angle from surface normal, to observer. 0 .. pi. THIS MATCHES
        
        (trgepc, srfvec, phase, solar, emmisn) = sp.ilumin('ELLIPSOID', 'PLUTO', et, 'IAU_PLUTO', abcorr, 'New Horizons', vec_pluto_psurf_iau)
        ang_pluto_sc_2[j,i] = emmisn  # Angle from surface normal, to observer. 0 .. pi. THIS MATCHES
        
# Make some diagnostic plots
                 
num_cols = 1
num_rows = 3

ang_vis = math.pi/2

# These 'ang_' arrays go from 0 .. pi. This makes sense. 
# The sub-solar position should have an angle of 0. The Anti-solar position should have angle pi.

plt.subplot(num_rows, num_cols, 1) 
plt.imshow(ang_pluto_sun.T < ang_vis, origin='lower')
plt.title('Sun visible from Pluto')

plt.subplot(num_rows, num_cols, 2) 
plt.imshow(ang_pluto_charon.T < ang_vis, origin='lower')
plt.title('Charon visible from Pluto')

plt.subplot(num_rows, num_cols, 3) 
plt.imshow(ang_pluto_sc.T < ang_vis, origin='lower')
plt.title('NH visible from Pluto')

plt.tight_layout()

plt.show()                                          

# Now, once we have validated all the angles, make a plot of Pluto in the sky. 
# Plot the RA/Dec of each of these Lon/Lat points, along with the visibility.
# Do this for all three objects.

# Validate the angles.
# Calculate the sub-SC Lon/Lat, and the sub-solar Lon/Lat.
# We can compare these to GV's results.
# Search for the position where these are closest to zero.

ang_sub_sc = np.min(ang_pluto_sc)
w = np.where(ang_pluto_sc == ang_sub_sc)

lon_sub_sc = lon[w[0]]
lat_sub_sc = lat[w[1]]

print(f'Sub-sc position on Pluto: lon lat = {lon_sub_sc}, {lat_sub_sc}')

ang_sub_sun = np.min(ang_pluto_sun)
w = np.where(ang_pluto_sun == ang_sub_sun)

lon_sub_sun = lon[w[0]]
lat_sub_sun = lat[w[1]]

print(f'Sub-sun position on Pluto: lon lat = {lon_sub_sun}, {lat_sub_sun}')

# Double-check: this below should be the sub-solar point:
# I got these positions from GV. GV uses LHR (old-style coords).

lon_rad_k = 265.39 * d2r
lat_rad_k = -51.56 * d2r

vec_pluto_psurf = sp.latrec(radius_pluto, lon_rad_k, lat_rad_k) # From P_center to P_surface

vec_psurf_sun = vec_pluto_sun - vec_pluto_psurf
ang_pluto_sun_k = sp.vsep(vec_psurf_sun, vec_pluto_psurf)

print(f'At lon={lon_rad_k*r2d} deg, lat={lat_rad_k*r2d} deg, ' + 
      f'solar angle = {ang_pluto_sun_k * r2d} deg')

# Now, go thru and convert each of these lon/lat points on Pluto, to an RA/Dec as seen from S/C.

ra_disc    = np.zeros([num_lon, num_lat])
dec_disc   = np.zeros([num_lon, num_lat])

for i,lat_i in enumerate(lat_rad):
    for j,lon_j in enumerate(lon_rad):        
        
        vec_pluto_psurf_iau   = sp.latrec(radius_pluto, lon_j, lat_i) # From P_center to P_surface
        vec_pluto_psurf = np.dot(mx, vec_pluto_psurf_iau) # Convert to J2000

        vec_sc_psurf = -vec_pluto_sc + vec_pluto_psurf
        (_, ra, dec) = sp.recrad(vec_sc_psurf)
        ra_disc[j,i] = ra
        dec_disc[j,i] = dec
        
ra_disc = np.array(ra_disc)
dec_disc = np.array(dec_disc)

# Calc if Pluto is visible to spacecraft

is_vis_pluto = ang_pluto_sc < (90 * d2r)

###########
# Now make a final plot of the entire disk, with regions marked
###########

#%%%

ms = 5  # Set the marker size

plt.plot(ra_disc*r2d, dec_disc*r2d, 
         linestyle='none', marker='.', ms=ms, color='black')


# Plot if Charon center is visible   

is_vis_charon = ang_pluto_charon < (90 * d2r)
is_good = np.logical_and(is_vis_pluto, is_vis_charon)
plt.plot(ra_disc[is_good]*r2d, dec_disc[is_good]*r2d, 
         linestyle='none', marker='.', ms=ms, color='grey')

# Plot the lat=0 line. Q: Why is this only getting plotted halfway?

is_good = np.logical_and(is_vis_pluto, np.abs(lat_2d) < 1)
# is_good = np.abs(lat_2d) < 10

plt.plot(ra_disc[is_good]*r2d, dec_disc[is_good]*r2d, 
         linestyle='none', marker='.', ms=1, color='blue')

# Plot the lon=0 line

is_good = np.logical_and(is_vis_pluto, np.abs(lon_2d) < 1)
plt.plot(ra_disc[is_good]*r2d, dec_disc[is_good]*r2d, 
         linestyle='none', marker='.', ms=1, color='lightblue')

# Plot if Sun is visible (i.e., emission angle from normal is < 90 deg)

is_vis_sun = ang_pluto_sun < (90 * d2r)
is_good = np.logical_and(is_vis_pluto, is_vis_sun)
plt.plot(ra_disc[is_good]*r2d, dec_disc[is_good]*r2d, 
         linestyle='none', marker='.', ms=ms, color='yellow')

plt.title(ut)
plt.xlabel('RA [deg]')
plt.ylabel('Dec [deg]')
plt.xlim([91.82, 91.6])
plt.gca().set_aspect('equal')
plt.show()
