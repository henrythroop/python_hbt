#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 08:47:27 2017

@author: throop
"""

# Process photo2web captions files 

import os
import glob
from subprocess import call  # Best way to call a function from python
from subprocess import check_output # Call, and return the output

# 
# 

file_captions = 'captions.txt'
dir_originals = 'originals'
dir_show      = '/Users/throop/photos/Trips/Test'
dir_originals = os.path.join(dir, 'originals', dir_show)

# With no arguments, take all of the files in the 'originals' dir, and one-by-one process them.
# Except if originals doesn't exist, then use the current directory.

if (os.path.isfile(dir_originals)):
    dir = dir_originals
else:
    dir = dir_show

images = [glob.glob(dir + "/*.jpg"), 
          glob.glob(dir + "/*.jpeg"),
          glob.glob(dir + "/*.JPG"),
          glob.glob(dir + "/*.JPEG")]

images = [item for sublist in images for item in sublist]

# Read the existing captions file
# It is 2 * N lines long -- with newlines separating each entry

#lun = open(os.path.join(dir, file_captions), 'r') # 'r' = read
#captions = lun.readlines()
#lun.close()

# Read all the captions from JPEG files

captions_new = []

for image in images:
            # Read the output, and then convert from byte into string
    caption = check_output(['exiftool', '-Description', image]).decode("utf-8")
    caption = caption[34:]  # Remove first few bytes from it
    
    captions_new.append(caption)

captions = captions_new

# Write all captions to the output file

lun = open(os.path.join(dir, file_captions), 'w')
for caption in captions:
    lun.write(caption)  # Damn. Gives error of 'must be str, not bytes'
    lun.writeline("\n")
lun.close()
print("Wrote: " + file_captions)


# .extend(glob.glob("*.JPG")).extend(glob.glob("*.jpeg")).extend(glob.glob("*.JPEG"))
    
# With multiple arguments:
#   Read the captions.txt
#   Read the list of files
#   Process the appropriate one (or more)
#   Write out the file
   
# First see if we have a 