#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 20 21:11:35 2018

@author: throop
"""

import glob
import os.path

dir_photos = '/Users/throop/photos/Trips/Test'

files_original = glob.glob(os.path.join(dir_photos, 'originals/*.jpg'))

file_captions = os.path.join(dir_photos, 'captions.txt')
file_header   = os.path.join(dir_photos, 'header.html')
file_footer   = os.path.join(dir_photos, 'footer.html')
file_out      = os.path.join(dir_photos, 'show.html')

captions = []
header   = []
footer   = []

with open(file_captions, "r") as ins:
    captions = []
    for line in ins:
        line = line.replace('\n', '')  # Kill blank lines (which are intentional separators), and strip newlines too.
        if line:
          captions.append(line)

# Read header
          
with open(file_header, "r") as lun:
    for line in lun:
        header.append(line)

# Read footer
        
with open(file_footer, "r") as lun:
    for line in lun:
        footer.append(line)

# Now do the output
        
lun = open(file_out, "w")

for line in header:
    lun.write(line)

lun.write('<div class="demo-gallery">\n')
lun.write('<ul id="lightgallery" class="list-unstyled row">' + "\n")

for i,file in enumerate(files_original):

    if  '##' in captions[i]:
        (caption, section) = captions[i].split('##')
        captions[i] = caption
#        lun.write('</ul>')
#        lun.write('\n')
#        lun.write("<hr>\n")
        lun.write(f'{section}\n')
#        lun.write('<ul id="lightgallery" class="list-unstyled row">' + "\n")
  
    line = (' <li \n' + 
#    line = (' <li class="col-xs-6 col-sm-4 col-md-3"\n' + 
		  f'     data-src="originals/{os.path.basename(file)}"\n' +  
		  f'     data-sub-html="<h2>{captions[i]}</h4>">\n' + 
#           f'      <a href="originals/{os.path.basename(file)}"' + 
            '>' + 
           f'      <a href="">' + 
           f'<img src="thumbnails/s{os.path.basename(file)}">' + 
            ' </a>\n' + 
            ' </li>\n\n')

    
    lun.write(line)
    
lun.write('</ul>\n')    
lun.write('</div>\n')
    
for line in footer:
    lun.write(line)    
    
lun.close()
print(f'Wrote: {file_out}')
