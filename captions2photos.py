#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 00:42:59 2021

@author: throop
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 20 21:11:35 2018

@author: throop

CAPTIONS2PHOTOS.PY

This reads a captions.txt file, and applies it to a list of images.

This idea is that this can be used to bring a very old photo2web gallery up to 'modern' standards.
Original version of photo2web used a 'captions.txt' and 'sections.txt' file. New versions of this 
(since 2010 or so) embed the caption + section in the EXIF info for the file.

No inputs. It works in the current directory.

HBT 16-Jan-2021.

"""


import glob
import os.path
import os
from html import escape
from bs4 import BeautifulSoup  # HTML parser
import subprocess
import datetime
from shutil import copyfile

def captions2photos():
    
    # Get into the right directory
    
    os.chdir('/Users/throop/photos/Trips/NH_Launch_Jan06_2')
    
    # Read the list of image files
    
    files = sorted(glob.glob('*.jpg'))
    print(f'{len(files)} images found')
    
    # Read the captions
    
    f = open('captions.txt', 'r')
    data = f.read()
    captions = data.split("\n\n")
    if captions[-1] == '':
      captions = captions[0:-1]   # Chop the final line, if it's blank
    
    # Read the list of sections
    
    f = open('sections.txt', 'r')
    data = f.read()
    sections = data.split("\n\n")
    if sections[-1] == '':
      sections = sections[0:-1]
    
    # Make a dictionary for the sections
    
    sections_dict = {}
    for section in sections:
        (num, sect) = section.split(' ',1)
        sections_dict[int(num)] = sect
    
    # Now loop over them all
    
    for i, caption in enumerate(captions):  # This starts at 1
      if (files[i] == captions[i]):
          print(f'N={i}: Skipping, matched')
      else:    
          print(f'N={i}, file={files[i]}\n')
          print(f'{captions[i]}')

          cap = captions[i]
          
          # If this is a new section for this image, print a tag for it
          
          if i in sections_dict.keys():  
              cap = cap + '## ' + sections_dict[i]
              print(f'Identified section with cap = {cap}')
              print(f'{files[i]}')

# Now actually set the caption in the jpeg file.
# Note that this line works, but only once! After an image caption has been set, sips will silently fail
# on any additional attempts.
          
          out = subprocess.check_output(['sips', '--setProperty', 'description', cap, files[i]])

      print('-------')
    

# =============================================================================
# Call the main code
# =============================================================================

if (__name__ == '__main__'):
    captions2photos()
    
