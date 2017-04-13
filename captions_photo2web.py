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
import sys  # For getting sys.argv

# 
# 

file_captions = 'captions.txt'
dir_originals = 'originals'
dir_show      = '/Users/throop/photos/Trips/Test'
dir_originals = os.path.join(dir, 'originals', dir_show)

#==============================================================================
# Read the captions.txt file
#==============================================================================
def get_captions_from_file(file):
    """ 
    Return an array with all the captions in a captions.txt file, one per entry
    """     
    lun = open(file, 'r') # 'r' = read
    captions_raw = lun.readlines()
    lun.close()
    captions = []
    for i in range(int(len(captions_raw)/2)):
        captions.append(captions_raw[i*2])
    return captions

    
#==============================================================================
# Write all captions to the output file
#==============================================================================

def write_captions_to_file(file, captions):

    lun = open(os.path.join(dir, file_captions), 'w')
    for caption in captions:
        lun.write(caption + "\n")  
    #    lun.writeline("\n")
    lun.close()
    print("Wrote: " + file_captions)

#==============================================================================
# Read all the captions from JPEG files
#==============================================================================

def get_captions_from_images(files):
 
    """
    Return captions for all the files listed by name.
    """
    captions_new = []
    
    for file in files:
                # Read the output, and then convert from bytes into string
        caption = check_output(['exiftool', '-Description', file]).decode("utf-8")
        caption = caption[34:]  # Remove first few bytes from it
        
        captions_new.append(caption)
    
    captions = captions_new

    return captions

# Process commandline arguments

if (len(sys.argv) > 1):
    index_image = sys.argv[1]
    
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

captions = get_captions_from_file(os.path.join(dir, file_captions))

# If 
# .extend(glob.glob("*.JPG")).extend(glob.glob("*.jpeg")).extend(glob.glob("*.JPEG"))
    
# With multiple arguments:
#   Read the captions.txt
#   Read the list of files
#   Process the appropriate one (or more)
#   Write out the file
   
# First see if we have a 