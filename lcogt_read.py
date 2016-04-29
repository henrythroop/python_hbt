# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 12:00:21 2014
# Program reads all of the SALT dataset and makes sense of it.
# Makes plot of which images were taken when
# Creates a post-facto 'observing log' for the SALT data
# Doesn't do any data analysis at all.
#
# HBT Aug-2014

@author: throop
"""

import pdb
import glob
import os.path
from   subprocess import call
import astropy
from   astropy.io import fits
import matplotlib.pyplot as plt # pyplot
import numpy as np
from   pylab import *  # So I can change plot size: 
                     # http://stackoverflow.com/questions/332289/how-do-you-change-the-size-of-figures-drawn-with-matplotlib

dir_data = "/Users/throop/Dropbox/data/LCOGT Comet Oct14"

# Get the full list of files

file_list = glob.glob(dir_data + '/*fits')
files = np.array(file_list)

# Read the JD from each file. Then sort the files based on JD.

jd = []
for file in files:
    hdulist = fits.open(file)
    jd.append(hdulist[0].header['MJD-OBS']) # Modified JD (with some offset) 
    hdulist.close()
    
indices = np.argsort(jd)
files = files[indices] # Resort by JD

#SITE
#TELESCOP
#LATITUDE
#LONGITUD
#OBSTYPE
#EXPTIME
#FILTER1
#FILTERI1
#FILTER2
#FILTERI2
#FILTER3
#FILTERI3
#FILTER
#INSTRUME
#INSTYPE # SBIG o sinistro
#GAIN
#AIRMASS
#MOONSTAT # Up/Down
#MOONDIST # Distance in degrees
#MOONFRAC # fraction illuminated
#SECPIX # arcsec/pixel
#SUNALT
#SUNDIST
#NAXIS1 # dx
#NAXIX2 # dy
#OBJECT # Target name
#RA
#DEC
#AZIMUTH
#ALTITUDE
 
naxis1  = [] # x dim of data. ** Before or after summing??
naxis2 =  [] # y dim of data
ra     =  [] # RA
dec    =  [] # Dec
azimuth = []
altitude =[]
filter =  [] # "I", "V", etc.
filter1=  [] # Air
filteri1= [] # Air
filter2=  [] # Air
filteri2= [] # Air
filter3=  [] # 'R', 'I', etc. But not Rs, Ic, etc.
filteri3= [] # 'COUS-RS-179', etc. The actual filter catalog number
airmass = [] # airmass
site    = [] # which site
telescop= [] # which scope
exptime = [] # seconds
object  = [] #
date_obs= [] #
latitude= [] # of telescope
longitud= [] # of telescope
date_obs= [] # starting time of exposure
mjd_obs = [] 
obstype = [] # 'exposure'
moonstat =[] # up/down
moondist =[] # distance in degrees
moonfrac =[] # fraction, 0 .. 1  
sunalt  = [] # solar altitude, degrees
sundist = [] # solar distance, degrees 
instrume= [] # long detailed name of the instrument
instype = [] # SBIG o sinistro 
gain    = [] # gain
ccdsum  = [] # '2 2' -> 2x2 bininng 
secpix  = [] # arcsec/pixel. 'Nominal' -> before binning, I think.

# print hdulist[0].header to display the whole thing
# if OBSMODE = IMAGING or INSTRUME=SALTICAM, then no gratings.

files_short = np.array(files)
for i in range(files.size):
    files_short = files[i].split('/')[-1]  # Get just the filename itself

for file in files:
    print "Reading file " + file

# Open the file
    hdulist = fits.open(file)

# Read the header

    header = hdulist[0].header

# Get a list of all the fields in the header. (We don't actually use this list.)

    keys = header.keys()

# Now go and grab each one

    naxis1      = np.append(naxis1, header['NAXIS1'])
    naxis2      = np.append(naxis2, header['NAXIS2'])
    ra          = np.append(ra, header['RA'])
    dec         = np.append(dec, header['DEC'])
    altitude    = np.append(altitude, header['ALTITUDE'])
    azimuth     = np.append(azimuth, header['AZIMUTH'])
    filter      = np.append(filter, header['FILTER'])
    filter1     = np.append(filter1, header['FILTER1'])
    filter2     = np.append(filter2, header['FILTER2'])
    filter3     = np.append(filter3, header['FILTER3'])
    filteri1    = np.append(filteri1, header['FILTERI1'])
    filteri2    = np.append(filteri2, header['FILTERI2'])
    filteri3    = np.append(filteri3, header['FILTERI3'])
    site        = np.append(site, header['SITE'])
    airmass     = np.append(airmass, header['AIRMASS'])
    telescop    = np.append(telescop, header['TELESCOP'])
    instrume    = np.append(instrume, header['INSTRUME'])
    instype     = np.append(instype, header['INSTYPE'])
    exptime     = np.append(exptime, header['EXPTIME'])
    object      = np.append(object, header['OBJECT'])
    date_obs    = np.append(date_obs, header['DATE-OBS'])
    mjd_obs     = np.append(mjd_obs, header['MJD-OBS'])
    latitude    = np.append(latitude, header['LATITUDE'])
    longitud    = np.append(longitud, header['LONGITUD'])
    obstype     = np.append(obstype, header['OBSTYPE'])
    moonstat    = np.append(moonstat, header['MOONSTAT'])
    moondist    = np.append(moondist, header['MOONDIST'])
    moonfrac    = np.append(moonfrac, header['MOONFRAC'])
    sundist     = np.append(sundist, header['SUNDIST'])
    sunalt      = np.append(sunalt, header['SUNALT'])
    gain        = np.append(gain, header['GAIN'])
    ccdsum      = np.append(ccdsum, header['CCDSUM'])

    hdulist.close() # Close the FITS file

# Make a plot of altitude vs. date

jd0 = min(mjd_obs)
plot(mjd_obs - jd0, altitude, linewidth=0, color='green', marker='*')
plt.title('LCOGT Data')
plt.xlabel('JD - ' + repr(jd0))
plt.ylabel('Altitude')
plt.show()

# Make a table of all of the data

# Now plot some images

dx_extract = 500
dy_extract = 500

for file in files:
  print "Opening image: " + file

  hdulist = fits.open(file)
  image = hdulist[0]
  hdulist.info()   # Print info about this file

  a = hdulist[0].data   # Read the file
  
  plt.set_cmap('gray')
#   plt.imshow(log(a))
  
  dx = a.shape[0]
  dy = a.shape[1]

# Extract part of the image.
# Remember that python images are vertical, then horizontal!

  a_center = a[(dx-dx_extract)/2 : (dx+dx_extract)/2, (dy-dy_extract)/2:(dy+dy_extract)/2]
  plt.imshow(log(a_center))
  plt.title(file)
  plt.show()

# print "Done reading all files"

rcParams['figure.figsize'] = 8, 6 # Make plot normal size

files_short = []
for file in files:
    files_short.append(file.replace(dir_data + '/', '').replace('.fits',''))
    
# Convert some things to numpy arrays. Is there any disadvantage to this?

instype_short = empty_like(instype)
for i in range(instype.size):
    instype_short[i] = instype[i].split('-')[2]

for i in range(ccdsum.size):
    ccdsum[i] = ccdsum[i].replace(' ', 'x')    

is_pair_prev = empty_like(moonstat, dtype=bool)

date_short = empty_like(date)
for i in range(date_short.size):
    date_short[i] = (date[i].replace('T', ' '))[0:19]
    
# How much of an image to extract?

dx_sub = 300
dy_sub = 300

num_files = files.size

sub = np.zeros((num_files, dx_sub, dy_sub)) 
   
plt.set_cmap('gray')
   
# Now display thumbnails of all frames
for i in range(len(files)):
    hdulist = fits.open(files[i])
    data = hdulist[0].data
    sub[i,:,:] = data[(naxis1[i] - dy_sub)/2:(naxis1[i] + dy_sub)/2, (naxis1[i] - dx_sub)/2:(naxis1[i] + dx_sub)/2]
#    plt.imshow(log(sub[i]))
#    plt.title('Filter ' + filter_long[i] + ', ' + instype_short[i] + ', ' + repr(exptime[i]) + ' sec, ' + ccdsum[i] + ', ' + date_short[i] )
    is_pair_prev[i] = (exptime[i] == exptime[i-1]) & (filter_long[i] == filter_long[i-1]) & (telescope[i] == telescope[i-1])
    hdulist.close()

for i in range(len(files)):   
    if (is_pair_prev[i]):
        print 'image ' + repr(i-1) + '+' + repr(i)
        title = 'Filter ' + filter_long[i] + ', ' + instype_short[i] + ', ' + repr(exptime[i]).replace('.0', '') + ' sec x2, ' +  ccdsum[i] + ', ' + date_short[i] 

        plt.imshow(log(sub[i] + sub[i-1]))
        plt.title(title)

        file_out = "/Users/throop/Dropbox/data/LCOGT Comet Oct14/Out/image" + repr(i).replace('.0', '') + '.png'
        plt.savefig(file_out)
        print 'Wrote: ' + file_out
        
        show()
        if (i == 20) | (i == 18) | (i == 16) | (i == 14): 
            x = 140
            y = 180
        if (i == 12) | (i == 10) | (i == 8) | (i == 6):
            x = 100
            y = 157
        if (i == 3):
            x = 120
            y = 160
        if (i == 1):
            x = 120
            y = 250
        sub_sub = sub[i][y-30:y+30, x-30:x+30]
        dn_max = repr(round(np.amax(sub_sub))).replace('.0', '')
        plt.imshow(log(sub_sub),interpolation='none')
        plt.title(title + ', DN_MAX = ' + dn_max)
        file_out = "/Users/throop/Dropbox/data/LCOGT Comet Oct14/Out/image" + repr(i).replace('.0', '') + '_zoom.png'
        plt.savefig(file_out)
        print 'Wrote: ' + file_out
        show()
        

