# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 22:56:46 2016

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

#########
# This is just an example to create and plot the backplane for one image
##########
    
file = '/Users/throop/Dropbox/Data/NH_Jring/data/jupiter/level2/lor/all/lor_0034676944_0x630_sci_1.fit'
#file = '/Users/throop/Dropbox/Data/NH_Jring/data/jupiter/level2/lor/all/lor_0034676552_0x630_sci_1.fit'

plane = hbt.create_backplane(file)

arr_image = hbt.read_lorri(file)

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
