# -*- coding: utf-8 -*-
"""
READ_ALICE.PY

Created on Wed Jul 13 13:48:10 2016

@author: throop

"""

##########
# Read an Alice FITS file from disk
##########

import astropy
from   astropy.io import fits
import numpy as np
import glob
import hbt # Seems kind of weird to have to import the module to which this function belongs...
                
def read_alice(file, frac_clip=0.9, polyfit=True, bg_method='None', bg_argument=4, autozoom=False):
    """    
    Reads an FITS file from disk. Returns via dictionary: 'spect', 'alam', 'count_rate', 'error'
    """

    dir_images = '/Users/throop/data/NH_Alice_Ring/data/pluto/level2/ali/all'
    
# If there is an empty filename passed, just return an array of random noise, but right size    
    if (file == ''):
        return np.random.random(size=(1024,1024))

# If filename is just an MET (either string or int), then look up the rest of it.

#    file = '0299391368'
    
    if hbt.is_number(file):
        file_list = glob.glob(dir_images + '/*{}*fit'.format(int(file)))
        
        if not file_list:
            print 'File not found'
            return 0
        
        file = file_list[0]
        
#    if ('hdulist') in locals():        # If there is already an hdulist, then close it. (Might double-close it, but that is OK.)
#        hdulist.close() 

# If filename has no '/' in it, then prepend the path

    if (dir_images.find('/') == -1):        
        hdulist = fits.open(dir_images + file)
        
    else:
        hdulist = fits.open(file)
        
    spect = hdulist['PRIMARY'].data            # 32           x 1024              Use hdulist.info() to get field list.
    alam  = hdulist['WAVELENGTH_IMAGE'].data   # 32 (spatial) x 1024 (wavelength)
    count_rate = hdulist['COUNT_RATE'].data    # 1894-element vector
    error = hdulist['ERROR_IMAGE'].data        # 32           x 1024
    header_spect = hdulist['PRIMARY'].header
    header_count_rate = hdulist['COUNT_RATE'].header
    
# Close the image

    hdulist.close()

# Assemble into a dictionary and return the results
#
# Fields to look at: 
#  header_spect[STARTMET, STOPMET, EXPTIME]
#  header_count_rate[SAMPLINT]

    out = {'spect' : spect, 'alam' : alam, 'count_rate' : count_rate, 'error' : error,
           'header_spect' : header_spect, 'header_count_rate' : header_count_rate}
    
    return out