#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 28 20:04:24 2017

This routine goes through the DPS Photo Archive (from Cruikshank and Morrison)
and runs OCR on the captions. There are 1200 images, so this is a lot better than 
retyping the captions. 

The OCR is not very good, and there are a ton of mistakes that need to be fixed manually, so it is still pretty slow.

OCR library = pytesseract. I installed tesseract using 'brew', which changed permissions on so many files that it may 
have made a mess of my filesystem. Therefore, I installed pytesseract only on desktop 'tomato'.
 
@author: throop
"""

import numpy as np
from PIL import Image
import pytesseract
import glob

list_subdirs = np.array(['1960s', '1970s', '1980s', '1990s', '2000s'])

list_subdirs = np.array(['1970s', '1980s', '1990s', '2000s'])

dir = "/Users/throop/photos/Trips/DPS_Archive/"

for subdir in list_subdirs:

    captions = []
 
    files = glob.glob(dir + subdir + '/*.jpg')

#    files = files[0:10]
    
    for file in files:
        
        file_short = file.split('/')[-1]
          
        caption = pytesseract.image_to_string(Image.open(file))
        
        print('{}: {}'.format(file_short, caption))

        captions.append(caption)

    file_out = dir + subdir + '/captions.txt'

    f = open(file_out, 'w')

    for i,file in enumerate(files):
        file_short = files[i].split('/')[-1]
        f.write("<" + file_short + ">\n")
        f.write(captions[i] + "\n")
        f.write('\n')
        f.write('---\n')
        f.write('\n')

    f.close()
    
    print("Wrote: " + file_out)
    
def other():
        
    file = dir + '1960s/1968 06.jpg'
    caption = pytesseract.image_to_string(Image.open(file))
    print(caption)
