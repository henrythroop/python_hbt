#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 20 21:11:35 2018

@author: throop

PHOTO2WEB2_INDEXER.PY

HBT 5-Feb-2021
                       
"""

import glob
import os.path
import os
from html import escape
from bs4 import BeautifulSoup  # HTML parser
import subprocess
import datetime
from shutil import copyfile

# =============================================================================
# Function definitions
# =============================================================================
   


# =============================================================================
# Start main code
# =============================================================================

def photo2web_process_hattenbach():
    """
    This is a one-off code that processes HH's family images, via photo2web.
    HH has 60 folders full of imags. This loops thru them all, reduces the size 
    of the absurdly large scans, and then runs photo2web on all of them.
    
    HH 7-Feb-2021


    """

    os.chdir('/Volumes/SSD External/Hattenbach_v2')
    
    dir_base = os.getcwd()
    
    dir_p2w = '/Users/throop/photos/Trips/'
    
    dirs = sorted(glob.glob(os.path.join(dir_base, '*')))
    
    quality_out = '60'
    size_out    = '2000x2000'
    
    for i,dir in enumerate(dirs):
        if os.path.isdir(dir):
            os.chdir(dir)
            dir_originals = os.path.join(dir, 'originals')
            dir_originals_fullres = os.path.join(dir, 'originals_fullres')

# For HH files, copy the 'actual' originals into a 'fullres' folder, for safekeeping

            if not os.path.isdir(dir_originals_fullres):
                os.rename(dir_originals, dir_originals_fullres)
                os.mkdir(dir_originals)
                
            files = glob.glob(os.path.join(dir_originals_fullres, '*'))

# Get a list of all the images

# For each image, make a low-res, low-quality image. This is just because the scanned files
# are huge and high-quality, and not useful for online. They are much larger than necessary. 
# So we use 'convert' to shrink them in size and quality, and put the output into 'originals' directory 
# for photo2web.

            for file in files:
                file_short = os.path.basename(file)
                file_in  = os.path.join(dir_originals_fullres,file_short)
                file_out = os.path.join(dir_originals,file_short)
                if not os.path.isfile(file_out):
                    cmd = (f'convert -resize {size_out} -quality {quality_out}' +
                           f' {file_in}' +
                           f' {file_out}')
                    print(f'{cmd}')
                    
                    subprocess.run(['convert', '-resize', size_out, '-quality', quality_out,
                                 file_in,
                                 file_out])

# Now, finally, go thru and do photo2web on all of them.
                
            print(f'\nProcessing directory {i}/{len(dirs)}  {dir}\n')
            subprocess.run(['cp', '-r', os.path.join(dir_p2w, 'header.txt'), '.'])
            subprocess.run(['cp', '-r', os.path.join(dir_p2w, 'photos.css'), '.'])
            if not os.path.exists('captions.txt'):
                subprocess.run(['captions_photo2web'])    
            subprocess.run(['photo2web_old'])
            subprocess.run(['photo2web'])

def photo2web_indexer():

    """
    This makes a very simple index.html page which lists *all* of the subdirs in a directory.
    For each of these subdirs, it makes a link, and puts up a few image thumbnails
    
    """
    
    import numpy as np
    
    os.chdir('/Volumes/SSD External/Hattenbach_v2')
    os.chdir('/Users/throop/photos/Trips')
    
    num_thumbs = 10    # Number of thumbnails per folder to display
    
    dir_base = os.getcwd()
        
    dirs = sorted(glob.glob(os.path.join(dir_base, '*')))
    
    # Create the output file
    
    file_out = 'index_all.html'
    lun   = open(file_out, "w")
    lun.write('<head><link rel="stylesheet" type="text/css" href="photos.css"></head>')
    
    
    for dir in dirs:
    
        if os.path.isdir(dir): 
            files = sorted(glob.glob(os.path.join(dir, 'thumbnails/s*jpg')))
            
            # If there are some thumbnails there, then proceed
            
            if len(files) > 0:
                basename = os.path.basename(dir)
                basename_edited = basename.replace('_', ' ')
                
                lun.write(f'<a href={basename}><h3>{basename_edited}</h3></a>\n')
                lun.write('<p>\n')
                print(dir)
                for i in range(np.amin([num_thumbs,len(files)])):
                    print(f'{i}/{num_thumbs} : {files[i]}')
                    lun.write(f'<a href={basename}><img src={basename}/thumbnails/{os.path.basename(files[i])}></a>\n')
                lun.write('<p>\n')
                print()
        
    lun.close()
    
        
# =============================================================================
# Call the main code
# =============================================================================

if (__name__ == '__main__'):
    photo2web_indexer()
    

