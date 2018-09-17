#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 08:47:27 2017

@author: throop
"""

# Process photo2web captions files
# Usage: 
#   captions_photo2web [number]
#
# With [number], extract that image number, with the first image being 1.
# With no argument, extracts all images in the 'originals' directory.
# This python routine replaces the perl script from c. 2008, and adds functionality
# to extract an individual caption.
#
# Format of the captions.txt file is that it has 2 x N lines, for N entries, separated by \n.
#
# THE CAPTIONS.TXT FILE CREATED HERE IS USED *ONLY* BY THE OLD HTML GALLERY CODE 
# -- NOT BY THE NEW JS GALLERY CODE!
#
# HBT 13-Apr-2017

import os
import glob
from   subprocess import check_output # Call, and return the output
import sys  # For getting sys.argv

file_captions = 'captions.txt'
dir_originals = 'originals'
dir_show      = os.path.curdir  # Directory that has the show.html file -- basically, root of the slideshow
dir_originals = os.path.join(dir_show, 'originals')  # 'original' subdir -- basically, directory of all the images

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
        caption = captions_raw[i*2]
#           caption = process_caption(caption)
        captions.append(caption)
    return captions

#==============================================================================
# Do some minimal processing on the caption, as needed
#==============================================================================

# Convert "YOUTUBE:TVzPkJSTYfo"
#  into   <iframe width="560" height="315" src="https://www.youtube.com/embed/ENM0-7R7jMw" 
#         frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>
#
# This is for the old-style 'long page of HTML' gallery, to make an embedded iframe.
# This code will still work OK if the caption has an <embed> youtube link.

def process_caption(caption):

    if 'YOUTUBE' in caption:  # This is case-sensitive
        caption = caption.replace('YOUTUBE:',  
	'\n<iframe width="560" height="315" src=https://www.youtube.com/embed/') + \
	'frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>'

        caption = caption.replace('\n', ' ').replace('\r', '') + '\n'

    return caption


#==============================================================================
# Write all captions to the output file
#==============================================================================

def write_captions_to_file(file, captions):

    lun = open(file_captions, 'w')
    for caption in captions:
        lun.write(caption + "\n")  
    #    lun.writeline("\n")
    lun.close()

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
        sys.stdout.write('.')      # Print progress bar to screen
        
        if (caption.strip() == ''):   # If there is no caption, use the filename itself
            caption = file.split('/')[-1] + "\n"
#            print("Using file = " + caption)

        caption = process_caption(caption)  # Do some processing on it, as needed (YOUTUBE)

        captions_new.append(caption)
#        print("Appended " + caption)
    
    print()  # Terminate the progress bar
    
    captions = captions_new

    return captions

#==============================================================================
# Print all captions to the screen
#==============================================================================

def print_captions(captions):
    """
    Diagnostic routine to print to screen, numbered
    """
    
    for i,caption in enumerate(captions):
        print("{}. {}".format(i, caption))
        
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

# Sort all the images alphabetically. They all have a simple name convention (001_, 002_,) so this is easy.

images.sort()

path_file_captions = os.path.join(dir_show, file_captions)

# If an image number was passed, read only that caption, and plug it back into the captions.txt file

if (len(sys.argv) > 1):
    index_image = int(sys.argv[1])
        
    # Read the existing captions file

    captions = get_captions_from_file(path_file_captions)
    
#    print_captions(captions)
    
    # Extract the caption from one file
    
    # Careful of indices here: To the human, the initial image is #1, but to python, it is #0.

    s = get_captions_from_images([images[index_image-1]]) # Wrap it into a list
    
    captions[index_image-1] = s[0]  # Result is returned as a one-item list. Extract it as a string.

    print("Extracting caption #{}".format(index_image))
    print(s[0])
    
# Otherwise, read all the captions, one per image.
 
else:
    
    captions = get_captions_from_images(images)
    
    print("Read {} captions.".format(len(captions)))

# Write the file out

#print_captions(captions)
    
write_captions_to_file(path_file_captions, captions)

print("Wrote: {}".format(path_file_captions))
