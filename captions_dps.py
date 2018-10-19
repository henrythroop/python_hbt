#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 15:41:03 2018

@author: throop
"""

# This routine makes captions for the DPS photo gallery.
# It essentially duplicates my earlier IDL routine that did the same.
# It requires a captions.txt file. That is, this current routine does not
# read the EXIF data. Reason for this is for editing a lot of captions (to shorten etc)
# it is faster to edit a text file, than import all into LR, edit, and re-export.
#
# HBT 18-Oct-2018 for Knoxville 50th DPS slideshow

import glob
import os

dir_base = '/Users/throop/photos/Trips/DPS18_Knoxville_Slideshow_Historic/'

cities = ['Cambridge 2005', 'Fajardo 2009', 'France 2011', 'Denver 2013', 'DC 2015', 'Pasadena 2016', 'Provo 2017']
# cities = ['Provo 2017']
# # cities = ['Fajardo 2009']
# # cities = ['Cambridge 2005']
# cities = ['Pasadena 2016']
# cities = ['France 2011']
# cities = ['Denver 2013']
# cities = ['DC 2015']

for city in cities:
    str_dir = 'DPS' + city.split(' ')[1][2:4] + '_' + city.split(' ')[0] # Create the directory name from the city + yr
 
    dir_gallery = os.path.join(dir_base, str_dir)
    file_captions = os.path.join(dir_gallery, 'captions.txt')
    
    with open(file_captions) as f:                           # Read all the captions
        lines = f.readlines()
            
    lines = [line.rstrip('\n') for line in open(file_captions)]
    captions = lines[::2]  # Skip every other one, which is blank!
    
    print(f'Read {len(captions)} captions from gallery {str_dir} of {city}')
    files = sorted(glob.glob(os.path.join(dir_gallery, '*jpg')))
    print(f'Read {len(files)} files from gallery {str_dir} of {city}')
    
    if (len(files) != len(captions)):
        exit(f'{len(files)} files and {len(captions)} captions!')
        
    for i,file_in in enumerate(files):                      # Loop over the files
        file_out = file_in.replace(dir_gallery, os.path.join(dir_gallery, 'captioned'))
        file_out = file_out.replace('.jpg', '_caption.jpg') # Ignore anything with .jpg in name
        if 'jpg' in captions[i]:
            captions[i] = ''
        caption = captions[i] + '   ' + city
    
    # Create the caption string. It can be done better than this to auto-size. The main problem is
    # that I am using -pointsize, which translates into a fixed # of pixels. I want instead to use a fixed
    # fraction of the vertical (or horizontal) dimension. So for huge files, then we get tiny captions.
    # See more at https://www.imagemagick.org/Usage/text/#caption
    
        str = f'convert {file_in} -gravity South -background black -stroke white -fill white -splice 0x50 ' +\
              f'-font Verdana -pointsize 40 -annotate +0+2 "{caption}" {file_out}'
        
        print(str.replace(dir_base, ''))
        os.system(str)    