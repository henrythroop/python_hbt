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

I tried to use justifiedGallery to make the thumbnails slightly more beautiful. I probably could have gotten it
working eventually, but I didn't, and it's only a minimal improvement.

Henry Throop 22-May-2018

To be done:
    - Make a new thumnail generator. Thumbs now need to be made in the perl code.
      Consider using sips, which might be faster than imagemagick.
    - Make page responsive to different screen sizes.
    - Figure out why thumbnail animations are broken. [Can't figure it out. Abandoning.]
    - Add some CSS or something to make the header information < full screen width [DONE]
    - Reduce caption width to less than full screen width.
    - Make a new caption extractor. Use 'sips --getProperty description *jpg', which is much faster
      than exiftool. [DONE]
    - Figure out what to do with YouTube links. [Ugh. I can embed them just fine, though ugly. But I can't 
      get them working the proper way.] [FIXED]
    - See if I can smallen the captions at all? Or put to side, or dismiss easily, or?? Since they always
      partially block.
    - Q: At one point I was getting a custom URL each time I clicked around. But no longer. Why not?? 
      I really prefer it. [SOLVED: Turn the 'hash' option back on.]

# 29-May-2018
      
      - Thumbnails are not yet made by this program. So, to use:
          - Output all files from LR
          - run 'photo2web_old' (perl). This will copy files into originals/, make the thumbnails, and extract captions,
                       and put into show_old.html and index_old.html.
          - run 'phtoto2web' (python, this program). This will make the HTML, fancy JS gallery, etc. and put into 
                       index.html .
"""

import glob
import os.path
import os
from html import escape
from bs4 import BeautifulSoup  # HTML parser
import subprocess
import datetime

# =============================================================================
# Function definitions
# =============================================================================

def get_all_captions(files):
    """
    Given a list of files, returns a list of captions.
    This uses 'sips', which is far faster than exiftool.
    It does not depend on using a captions.txt file.
    """
    
    captions_all = subprocess.check_output(['sips', '--getProperty', 'description'] + files).decode("utf-8")
    
    captions = []
    for i in range(len(files)):
        if (i < len(files)-1):
            pos1 = captions_all.find(files[i])
            pos2 = captions_all.find(files[i+1])
            caption_i = captions_all[pos1:pos2]
        else:
            pos1 = captions_all.find(files[i])
            caption_i = captions_all[pos1:]
        caption_i = caption_i.replace(files[i], '')
        caption_i = caption_i[16:]
        caption_i = caption_i[:-1]  # Remove final \n
        caption_i = caption_i.replace('<nil>', '')  # Remove <nil>
        captions.append(caption_i)
#        print(f'Added caption {caption_i}')
        
    return captions
        
def make_gallery_item(caption, basename, type = 'span'):
    """
    Return an HTML line for the gallery.
    
    Parameters
    -----
    
    caption:
        String caption, HTML.
        
    basename:
        String. Can be a filename (IMG_5534.jpg) or a URL (http://www.youtube.com/asdfk98)
        
    """
    
    if '.jpg' in basename:
        line  = f'<span class="item" data-sub-html="<span class=caption>{caption}</span>"' + \
                f' data-src="originals/{basename}">\n' + \
                f'  <a href="originals/{basename}"><img src="thumbnails/s{basename}"/></a>\n' + \
                f'  </span>\n\n'

    if 'youtu' in basename:
        id = basename.split('/')[-1]  # Get the video ID (e.g., wiIoy44Q4). # Create the YouTube thumbnail!
        line  = f'<span class="item"' + \
                f' data-src="{basename}"> ' + \
                f'  <a href="{basename}" data-src="{basename}">' + \
                f'  <img src="http://img.youtube.com/vi/{id}/default.jpg"/></a>' + \
                f'  </span>\n\n'

        # As a test, define the element as an <a> anchor.
        
#    line_a = f'<span class="item"> <a href="originals/{basename}"> <img src="thumbnails/s{basename}"/> </a></span>\n\n'

    return line

def make_thumbnails(files):
    """
    Make thumbnails for the specified files.
    """
    
    # OK, right now this code does nothing. I want it to do the same as the original Perl code did:
    
#    o Get a list of new .jpg's in the main directory
#    o For each one:
#        o If basename of it matches one in originals, make new thumbs and copy and replace
#        o If it doens't match an existing one, then (?? -- not sure) -- make new thumbs and copy, no rename
#       
    
    return None
    
# =============================================================================
# Start main code
# =============================================================================

def photo2web():

    dir_photos = os.getcwd()

    files_original = glob.glob(os.path.join(dir_photos, 'originals/*.jpg'))

    # Here, define a <span> </span> element which is the image itself. 
    # We tag this span with class=item, and then use a corresponding selector in the call to lightgallery.
    # This prevents lightgallery from being called on headers, <hr>, and random text on the page which is not pics.
    
    dir_js = '/Users/throop/photos/Trips/js'
    dir_lg = os.path.join(dir_js, 'lightGallery-master')
    
    file_captions  = os.path.join(dir_photos, 'captions.txt')   # List of all captions, one per line-pair, via exiftool
    file_header    = os.path.join(dir_js, 'header.html')    # Header with JS includes, CSS, etc.   
    file_footer    = os.path.join(dir_js, 'footer.html')  # HTML footer with JS startup, etc.
    file_header_txt= os.path.join(dir_photos, 'header.txt')  # Header file which I type manually. Line0 is gallery title
    file_out       = os.path.join(dir_photos, 'index.html')    # Final output filename
    
    captions = []
    header   = []
    footer   = []
    
    header_txt = []
    
    captions = get_all_captions(files_original)
    
    print(f'Read {len(captions)} captions')
    
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
    
    # Read HTML footer. Plug in the date as needed.
            
    datestr = datetime.datetime.now().strftime("%d %b %Y")
    with open(file_footer, "r") as lun:
        for line in lun:
            footer.append(line.replace('DATE_HERE', datestr))
    
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
    
    j = 0
    
    # Print a link to the old-style gallery

    lun.write("<p><a href=show_old.html><b>Slideshow (old-style, all photos on one long page)</b></a><br><p>\n")
    lun.write("<p><a href='..'><img src='../icons/info.gif' border=0><b>Return to list of galleries</b></a><p>\n")
    
    # Loop and print the entry for each image
    
    for i,file in enumerate(files_original):
    
        caption = captions[i]
    
        # If this image starts a new section, then create the HTML for that
    
        if  '##' in captions[i]:
            if (j > 0):
                lun.write('</div>\n')
            caption, section = caption.split('##')
            lun.write(f'<br><hr> <a name={j+1}> <h3>{section}</h3>\n\n')  # Anchor tag, so we can use index.html#1
            lun.write(f'<div id="gallery{j}">\n')
            j+=1
    
        # If caption is just a filename, then zero it out
        
        if '.jpg' in caption:
            caption = ''
            
        # Handle " < > etc in captions. But actually I'm not sure I want to do that... just quotes, perhaps.
        
        caption = caption.replace('"', '&quot;')
        caption = caption.replace("'", '&#x27;')
        
        basename = os.path.basename(file)

        # If the caption has a youtube link in it, make a new slide for that.
        # Convention is this: If there is a youtube movie, put its URL in the 
        # Lightroom caption for the *previous* image. Write it like this:
        #
        #   And here we are swimming.<a href=https://youtu.be/mov12498jE>Swimming movie!</a>
        #
        # Don't put in the <embed> or anything like that.
        
        if ('youtube.com' not in caption) and ('youtu.be' not in caption):
            line = make_gallery_item(caption, basename)                     # Normal caption and URL

        else:    
            matchstr        = '<a href=https://you'
            
            (caption1, html) = caption.split(matchstr)
            line1           = make_gallery_item(caption1, basename)
            html             = matchstr + html
            soup            = BeautifulSoup(html, 'html5lib')
            a               = soup.find_all('a')[0]
            url             = a.get('href')
            caption2        = a.contents[0]
            line2           = make_gallery_item(caption2, url)            
            line            = line1 + line2
        
        # Print the entire HTML line, with image, thumbnail, and caption, to the file
    
        lun.write(line)
#        print(line) 
       
   # Print the HTML footer, and close the file
    
    lun.write('</div>\n')    
    lun.write('</div>\n')
        
    for line in footer:
        lun.write(line)    
        
    lun.close()
    print(f'Wrote: {file_out}')

# =============================================================================
# Call the main code
# =============================================================================

if (__name__ == '__main__'):
    photo2web()
    
