# -*- coding: utf-8 -*-
"""
Process Tracking Sheet Timelapse

"""

import glob
import os.path
import os
from   html import escape
from   bs4 import BeautifulSoup  # HTML parser
import subprocess
import datetime

from PIL import Image 
from PIL.ImageStat import Stat

dir_images_in = '/Users/throop/Desktop/ssw_tracking/in/'
list_files = sorted(glob.glob(dir_images_in + '*'))

dir_out = '/Users/throop/Desktop/ssw_tracking_out'

# list_files = list_files[0:10]
i = 0
sums = []

# Loop over all the files in the directory

for file in list_files:

    img = Image.open(file) 
    width,height = img.size

# Crop as needed
    
    img_c = img.crop((790, 400, 2300, 1200)) # left top right bottom
    img_c.show()

    file_out = file.replace('/in/', f'/out/{i:05}_')

# Get a checksum for the cropped image (or just a 'sum', which is just as good)
    
    sum = Stat(img_c).sum
    sums.append(sum)
    i+=1

# If the cropped image's checksum is the same as the previous one, then skip it
# And if it *has* changed, write the output image
    
    if (i > 1):
        if sum == sums[i-2]:
            print('sum matched! -- skipping write')
        else:   
            img_c.save(file_out)
            print(f'Wrote: {file_out}')