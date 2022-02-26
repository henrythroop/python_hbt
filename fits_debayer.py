#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 23:41:57 2021

@author: throop
"""

# from hbt_short import hbt 

# import hbt_short

import glob
import os.path
import os
from html import escape
from bs4 import BeautifulSoup  # HTML parser
import subprocess
import datetime
from shutil import copyfile
import matplotlib.pyplot as plt
import math
import numpy as np
import astropy
import astropy.modeling
import skimage.transform as skt  # This 'resize' function is more useful than np's
#import matplotlib.pyplot as plt # Define in each function individually
import matplotlib                # We need this to access 'rc'
import spiceypy as sp
from   astropy.io import fits
import subprocess
import hbt
import warnings
import importlib  # So I can do importlib.reload(module)
from   photutils import DAOStarFinder
import photutils
from   astropy import units as u           # Units library
import scipy.ndimage
from astropy.time import Time

from   scipy.optimize import curve_fit
from   astropy.convolution import Box1DKernel, Gaussian1DKernel, convolve


def debayer(im, cell=0):

# Returns an array which is binned 2x2 from the input matrix.

# if cell = {1,2,3,4}, then extract just that cell.
# If cell = 0, then add all four cells.

    a = im[0::2, 0::2]
    b = im[0::2, 1::2]
    c = im[1::2, 0::2]
    d = im[1::2, 1::2]
    
    im_out = a + b + c + d
    
    if (cell == 1):
        im_out = a
    if (cell == 2):
        im_out = b
    if (cell == 3):
        im_out = c
    if (cell == 4):
        im_out = d
        
    return(im_out)

def debayer_all():
    
    path_in = '/Users/throop/Downloads/mrjmcr/Kalliope/raw/'
    path_out = path_in.replace('/raw', '/debayer')
    
    files = glob.glob(os.path.join(path_in, '*.fits'))
    files = np.sort(files)
    
    file = files[0]
    
    cell = 4
    
    for file in files:
        
        # Read the original data + header
        
        hdu = fits.open(file)
        img = hdu['PRIMARY'].data
        header = hdu['PRIMARY'].header
        hdu.close()
        
        # Remove the stripes
        
        img_d = debayer(img, cell=cell)
        
        stretch_percent = 95    
        stretch = astropy.visualization.PercentileInterval(stretch_percent)
           # PI(90) scales to 5th..95th %ile.    
        
        # Plot it
        
#        plt.imshow(stretch(img_d), origin='lower')
#        plt.title(os.path.basename(file))
#        plt.show()

        if cell > 0:
            file_out = os.path.join(path_out,
               os.path.basename(file).replace('.fits', f'_{cell}.fits'))
        else:
            file_out = os.path.join(path_out, os.path.basename(file))
            
        # Write the new FITS file
        
        hdu_out = fits.PrimaryHDU(img_d, header=header)
        hdu_out.writeto(file_out, output_verify='ignore', overwrite=True)
        print(f'Wrote: {file_out}')
        

if (__name__ == '__main__'):
    debayer_all()
        

