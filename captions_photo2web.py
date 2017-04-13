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
dir_show      = os.path.curdir
dir_originals = os.path.join(dir_show, 'originals')

print("Path = {}".format(dir_originals))

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
        print(".", end='')
        captions_new.append(caption)
    
    captions = captions_new

    return captions

#==============================================================================
# MAIN CODE
#==============================================================================

# Process commandline arguments

if (len(sys.argv) > 1):
    index_image = sys.argv[1]
    
# With no arguments, take all of the files in the 'originals' dir, and one-by-one process them.
# Except if originals doesn't exist, then use the current directory.

if (os.path.exists(dir_originals)):
    dir = dir_originals
else:
    dir = dir_show

images = [glob.glob(os.path.join(dir, "*.jpg")), 
          glob.glob(os.path.join(dir, "*.jpeg")),
          glob.glob(os.path.join(dir, "*.JPG")),
          glob.glob(os.path.join(dir, "*.JPEG"))]

images = [item for sublist in images for item in sublist]

print("Found {} images in {} ".format(len(images), dir))

# Sort them. They all have a simple prefix (001_, 002_,) so this should be simple

images.sort()

path_file_captions = os.path.join(dir, file_captions)

if (len(sys.argv) > 1):
    index_image = int(sys.argv[1])
        
    # Read the existing captions file

    captions = get_captions_from_file(path_file_captions)
    
    # Extract the caption from one file

    s = get_captions_from_images([images[index_image]]) # WRap it into a list
    
    captions[index_image] = s[0]  # De-list it

    print("Extracting caption #{}".format(index_image))
    
    (path_file_captions, captions)

else:

    captions = get_captions_from_images(images)
    
    print("Read {} captions.".format(len(captions)))

# Write the file out
    
write_captions_to_file(path_file_captions, captions)

print("Wrote: {}".format(path_file_captions))

# If 
# .extend(glob.glob("*.JPG")).extend(glob.glob("*.jpeg")).extend(glob.glob("*.JPEG"))
    
# With multiple arguments:
#   Read the captions.txt
#   Read the list of files
#   Process the appropriate one (or more)
#   Write out the file
   
# First see if we have a 