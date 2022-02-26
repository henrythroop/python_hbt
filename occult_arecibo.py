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

# Routine to remove horizontal banding from QHY174 images. HBT 3-Jul-2021
# For 4337 Arecibo occultation data 

def destripe(im):
    
    rowmean      = np.median(im, 1) 
    rowmean[1:6] = 0
    
    arr_bias = np.add.outer(rowmean, 1+np.zeros(1920))
    
    im_out = im - arr_bias.astype(int)

    im_out = (im_out - np.amin(im_out)).astype(np.uint16)
          
    return(im_out)

def destripe_all():
    
    path = '/Users/throop/Data/Occultations/4337Arecibo_Jun21/01_49_08/'
    path = '/Users/throop/Data/Occultations/744Aguntina_Jul21/p1/'
    
    files = glob.glob(path + '*.fits')
    files = np.sort(files)
    
    file = files[0]
    
    for file in files:
        
        # Read the original data + header
        
        hdu = fits.open(file)
        img = hdu['PRIMARY'].data
        header = hdu['PRIMARY'].header
        hdu.close()
        
        # Remove the stripes
        
        img_d = destripe(img)
        
        stretch_percent = 95    
        stretch = astropy.visualization.PercentileInterval(stretch_percent) # PI(90) scales to 5th..95th %ile.    
        
        # Plot it
        
        # plt.imshow(stretch(img_d), origin='lower')
        # plt.title(file)
        # plt.show()
    
        file_out = file.replace('p1/', 'p1/cleaned/')
        
        # Write the new FITS file
        
        hdu_out = fits.PrimaryHDU(img_d, header=header)
        hdu_out.writeto(file_out, output_verify='ignore', overwrite=True)
        print(f'Wrote: {file_out}')
        
def print_timesteps_all():
    
    # Just a q&d function to print the dt between frames, so I can find dropped frames.
    
    path = '/Users/throop/Data/Occultations/4337Arecibo_Jun21/eVscope-wppu27/' # evscope
    path = '/Users/throop/Data/Occultations/4337Arecibo_Jun21/01_49_08/' # QHY
    path = '/Users/throop/Data/Occultations/744Aguntina_Jul21/p2/' # QHY
    path = '/Users/throop/Data/Occultations/4337Arecibo_Jun21/eVscope-wppu27/4337 Arecibo/' # eVscope, try 2
    
    files = glob.glob(path + '*.fits')
    files = np.sort(files)
    
    file = files[0]
    
    jd = np.zeros(len(files))
    date_obs = np.zeros(len(files)).astype('U50')
    
    for i,file in enumerate(files):
        
        # Read the original data + header
        
        hdu = fits.open(file)
        # img = hdu['PRIMARY'].data
        header = hdu['PRIMARY'].header
        date_obs[i] = header['DATE-OBS']  # This field name is used by both QHY and eVscope
        t = Time(date_obs[i], format='isot', scale='utc')
        jd[i] = t.jd
        if i > 3:
            djd_sec = (jd[i] - jd[i-1]) * 86400
            print(f'{i:5} {file} {date_obs[i]}  Sec since prev: {djd_sec:.2f}')
            if djd_sec < 0.1:
                print('Warning!!')
        hdu.close()
        
        # Remove the stripes
        
        # img_d = destripe(img)
        
        # stretch_percent = 95    
        # stretch = astropy.visualization.PercentileInterval(stretch_percent) # PI(90) scales to 5th..95th %ile.    
        
        # Grab the ET
        
        # plt.imshow(stretch(img_d), origin='lower')
        # plt.title(file)
        # plt.show()
    
        # file_out = file.replace('01_49_08/', '01_49_08/cleaned/')
        
        # Write the new FITS file
        
        # hdu_out = fits.PrimaryHDU(img_d, header=header)
        # hdu_out.writeto(file_out, output_verify='ignore', overwrite=True)
        # print(f'Wrote: {file_out}')   
    

