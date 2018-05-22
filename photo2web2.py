#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 20 21:11:35 2018

@author: throop

PHOTO2WEB2.PY

This is the Python implementation of Photo2Web. The only thing it shares with the original Perl version is the name,
and the file structure. The main difference is that the slides are generated by the JS 'photogallery' tool,
rather than individual HTML pages.

The caption font size, alignment, etc. are defined in lightgallery.css. To edit them,
put their new values into .lg-sub-html and its descendents in photos.css .

Henry Throop 22-May-2018

To be done:
    - Make a new thumnail generator. Thumbs now need to be made in the perl code.
      Consider using sips, which might be faster than imagemagick.
    - Make page responsive to different screen sizes.
    - Figure out why thumbnail animations are broken.
    - Add some CSS or something to make the header information < full screen width
    - Reduce caption width to less than full screen width.
    - Make a new caption extractor. Use 'sips --getProperty description *jpg', which is much faster
      than exiftool.

"""

import glob
import os.path
from html import escape

dir_photos = '/Users/throop/photos/Trips/Test'

files_original = glob.glob(os.path.join(dir_photos, 'originals/*.jpg'))

file_captions  = os.path.join(dir_photos, 'captions.txt')   # A list of all captions, one per line-pair, via exiftool
file_header    = os.path.join(dir_photos, 'header.html')    # Header with JS includes, CSS, etc.   
file_header_txt= os.path.join(dir_photos, 'header.txt')  # Header file which I type manually. First line is gallery title
file_footer    = os.path.join(dir_photos, 'footer.html')  # HTML footer with JS startup, etc.
file_out       = os.path.join(dir_photos, 'show.html')    # Final output filename

captions = []
header   = []
footer   = []

header_txt = []

with open(file_captions, "r") as ins:
    captions = []
    for line in ins:
        line = line.replace('\n', '')  # Kill blank lines (which are intentional separators), and strip newlines too.
        if line:
          captions.append(line)

# Read text header
        
with open(file_header_txt, "r") as lun:
    for line in lun:
        header_txt.append(line)

# Extract the title of the gallery from the header.txt file
        
title_gallery = header_txt[0]
header_txt = header_txt[1:]

# Read HTML header. Plug in the gallery name as needed.
          
with open(file_header, "r") as lun:
    for line in lun:
        header.append(line.replace('TITLE_HERE', title_gallery))

# Read HTML footer
        
with open(file_footer, "r") as lun:
    for line in lun:
        footer.append(line)

# Now do the output
        
lun = open(file_out, "w")

# Write HTML header

for line in header:
    lun.write(line)

# Write text header

for line in header_txt:
    lun.write(line)

lun.write('<div class="demo-gallery">\n')
lun.write('<div id="lightgallery" class="list-unstyled row">' + "\n")

# Loop and print the entry for each image

for i,file in enumerate(files_original):

    caption = captions[i]

    # If this image starts a new section, then create the HTML for that

    if  '##' in captions[i]:
        caption, section = caption.split('##')
        lun.write(f'<div><br><hr> <h3>{section}</h3> </div>\n\n')

    # If caption is just a filename, then zero it out
    
    if '.jpg' in caption:
        caption = ''
        
    # Handle " < > etc in captions. But actually I'm not sure I want to do that... just quotes, perhaps.
    
    caption = caption.replace('"', '&quot;')
    caption = caption.replace("'", '&#x27;')
    
    # Here, define a <span> </span> element which is the image itself. 
    # We tag this span with class=item, and then use a corresponding selector in the call to lightgallery.
    # This prevents lightgallery from being called on headers, <hr>, and random text on the page which is not pics.
    
    line_span = f'<span class="item" data-sub-html="<span class=caption>{caption}</span>"' + \
                f'data-src="originals/{os.path.basename(file)}">\n' + \
                f'  <img src="thumbnails/s{os.path.basename(file)}"/>\n' + \
                f'  </span>\n\n'
   
    # Finally, print the entire HTML line, with image, thumbnail, and caption, to the file

    lun.write(line_span)

# Print the HTML footer, and close the file

lun.write('</div>\n')    
lun.write('</div>\n')
    
for line in footer:
    lun.write(line)    
    
lun.close()
print(f'Wrote: {file_out}')

