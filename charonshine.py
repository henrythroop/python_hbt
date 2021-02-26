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

# print "Spice imported"
dir_sp = "/Volumes/SSD External"
file_tm = 'gv_kernels_new_horizons.txt'

os.chdir(dir_sp)

import math

r2d = 360 / (2 * math.pi)
d2r = 1/r2d

sp.furnsh(os.path.join(dir_sp, file_tm))

ut =  "2015 July 15 01:58:00"
et = sp.utc2et(ut)

# num_lonlat = 100
num_lat = 100
num_lon = 200

lon = np.linspace(-90,  90, num=num_lon)
lat = np.linspace(0,   360, num=num_lat)

lon_rad = lon * d2r
lat_rad = lat * r2d

ang_pluto_sun    = np.zeros([num_lon, num_lat])  # Angle from Pluto surface point to Sun, rad
ang_pluto_charon = np.zeros([num_lon, num_lat])  # Angle from Pluto surface point to Charon, rad
ang_pluto_sc      = np.zeros([num_lon, num_lat])  # Angle from Pluto surface point to Sun, rad

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

# (lon_2d, lat_2d) = np.meshgrid(lat,lon)

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
 
num_cols = 1
num_rows = 3

ang_vis = math.pi/2

# These 'ang_' arrays go from 0 .. pi. This makes sense. 
# The sub-solar position should have an angle of 0. The Anti-solar position should have angle pi.

# The sub-sc position should also.

plt.subplot(num_rows, num_cols, 1) 
plt.imshow(ang_pluto_sun.T < ang_vis, origin='lower')
plt.title('Sun')

plt.subplot(num_rows, num_cols, 2) 
plt.imshow(ang_pluto_charon.T < ang_vis, origin='lower')
plt.title('Charon')

plt.subplot(num_rows, num_cols, 3) 
plt.imshow(ang_pluto_sc.T < ang_vis, origin='lower')
plt.title('SC')

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
        vec_pluto_psurf = np.dot(mx, vec_pluto_psurf_iau) # COnvert to J2000

        vec_sc_psurf = -vec_pluto_sc + vec_pluto_psurf
        (_, ra, dec) = sp.recrad(vec_sc_psurf)
        ra_disc[j,i] = ra
        dec_disc[j,i] = dec
        
ra_disc = np.array(ra_disc)
dec_disc = np.array(dec_disc)

is_good = ang_pluto_sun < 0.5
        
plt.plot(ra_disc[is_good], dec_disc[is_good], 
         linestyle='none', marker='.', ms=15)


              
