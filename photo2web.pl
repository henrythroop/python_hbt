#!/usr/bin/perl                   
# The first line defines the Perl Interpreter
# This file is *not* under version control.
#
# Written 01/28/02 by DPH
#
# Modified 20-Feb-02; 21-May-02 by HBT 
#                    To use header.txt file, apply timestamp, 
#                      and minor modifications to fonts & formattting.
#
# Modified 21-Aug-02 To change bg, font, etc. colors.
#
# Modified 21-Nov-02 To click on big image to get hires.
#                    Also fixed bug that </a> tags were missing from icons
#
# Modified 19-Feb-03 To add references to dimmed buttons on first, last images
#                    Also made references to icons dir relative rather than absolute, 
#                    so directory can be copied to CD, etc. more easily.
#                    Also removed e-mail address line
#
# Modified 13-Mar-03 Modified to make sure that mogrify only shrinks images,
#                     but doesn't enlarge them if they start small.
#
# Modified 16-Mar-03 Put all filename calls (in mogrify()) in quotes s.t. will
#                     work with files with spaces.
#                    Also modified to die() if # captions != # images 
#
# Modified 18-Mar-03 Creates thumbnail etc. images iff thumbnail doesn't exist,
#                     or it's older than the original jpeg.  Makes adding or modifying
#                     images orders of magnitude faster -- used to be all or nothing.
#                     An additional feature would be to remove old thumbnails too.
#
# Modified 29-Jun-03 Allows input of sections.txt file, which has format  
#                       "Image_number section_title" (separated by newline pairs).
#                    (where image_number starts at 1 for first image).
#
# Modified 30-Jun-03 Allows input of colors.txt file, which has a one-line entry
#                     such as 
#                       "color=white bgcolor=#aa00ff vlink=green"
#
# Modified  9-Jan-05 Allow for generation of an 'all-at-once' slideshow page, with 
#                     intermediate-sized images (order: thumbs, intermed, medium, full)
#
# Modified 25-Jan-05 Changed 'convert' flags from -geometry to -size/-resize, for 3x speed increase.
#
# Modified  8-Jun-05 If a caption is still the filename (or rather, has 'jpg' in it), then it is not printed.
#
# Modified 22-Aug-05 If file .photo2web_inhibit exists, then don't run here.
#
# Modified 20-Aug-07 If filename has 'Pano' in it, then it's a panorama -> make the thumbnail
#                     double its default size.
#                     [1-Jul-2013: Now the string is 'pan', not 'Pano']
#
# Modified 30-Jul-08 Exit if there's 0 photos in the directory.
#                    Fixed bug in which caption was ignroed if it contained a link to a jpg file.
#
# Modified  5-Oct-08 Add ALT=text to thumbnails.  Add #ANCHORS to dfft sections.
#
# Modified 15-Nov-08 Read and use photos.css, if it exists.  Also minor HTML changes.
#
# Modified 25-Mar-09 Added PREFETCH to the slides, to load next slide and image while waiting.
#
# Modified Apr-09    If file INHIBIT_PHOTO_INDEX exists, then no link to higher-up photos is made.
#
# Modified 30-Sep-09 Added option to supply a contact_info.txt file.
#
# Modified  5-Mar-09 Added option to put a link to RSS on index.html and show.html pages.
#                    Removed ALT=tag, since it got broken if I put too much fancy stuff in there, like YouTube embeds.
#                    I think it had to do with errors with quotes, so a better solution would be just to removed the 
#                    captions with quotes, or escape them.
#
# Modified 30-Mar-10 If no captions.txt file exists, then no reference is made to 'view with captions', etc.
#
# Modified 15-Apr-10 Reverted to previous behavior where in 'show.html' mode, clicking on photo now takes user not
#                    to the full-res image, but to the single-page slide for that pic.
#
# Modified 17-Jun-10 Only update a slide file if it's actually changed.
#
# Modified 7-Jul-13 Panorama file prints wider in the show.html file, not just the thumbnails. 
#                   To force this, just put 'pan' somwhere in the filename.
#
# Modified 27-Mar-14 More efficient handling of new versions of existing files exported from Lightroom.
#          o Put originals in a directory called 'originals'
#          o For each one moved there, check if there is an existing file there already with the same 
#             base filename (excluding the 3-digit prefix).
#             If there is, then put the existing prefix on the newly copied file, and move the new file to 'originals.'
#          o Thus, if 003_DSC_1224.jpg and originals/164_DSC_1224.jpg both exist, then former will be copied to the latter one.
#          o Point of this is so that if I re-export one file, I don't have to re-process all the files.
#
# Modified 1-Apr-2014 If the text $IMAGES is found in a caption or header, then that text is replaced with a link to the 
#            root images/ path (either .. or ../.., depending on if creating an individual slide, or the overall show page).
#
# Modified 1-Apr-2014 Header can now be multiple paragraphs. First one is for page title, and remainder are all put into html.
#
# Modified 20-Jan-2015 Section names are now read from captions directly (if there's no sections.txt file). This means less 
#             manual editing of sections.txt file necessary. Anything following ## IS A SECTION NAME.
#
# Modified 6-Jul-2016 show.html now includes a list of sections up top, and a quick jump-to link for each.
#
# Modified 6-Jul-2016 index.html (thumbnail view) now includes the descriptive text from header.txt file, as it probably should
#             have all along.
#
# Modified 10-Apr-2017 to allow vertical layout (that is, captions below pics), on the show.html page. This requires
#            commandline flags, which were not supported, but now are.
#
# Modified 17-Apr-2017 to call new python 'captions_photo2web' code, which allows for the extraction of individual captions, 
#            rather than all-at-once of previous version.
#
# Modified 28-May-2018 To output to files {index,show}_old.html. I am migrating to new python-based code, and will deprecate this one.
# Modified 6-Oct-2019. Changed link to 'thumbnails' to point to index.html, which is the new modern page, not the 'classic' 
#            thumbnail page index_old.html .
#
# Modified 3-Jan-2021. Fixed a bug that was not applying CSS properly to the index_old.html file.

# To do: o Make it put a timestamp & creator in the html code. [huh? done this for ages.]
#        o Make the large files optional (for saving space)
#        o Change the permissions on the files within the thumbnails/ dir s.t. 
#           they are guaranteed to be writeable (error if not)
#        o Make whole thing use CSS. [DONE]
#        o Add drop shadows to the thumbnails. [seems impossible using imagemagick.  Can 
#           probably force it using pre-computed shadows, but this fails for rotated images.]
#        o Only rewrite the SlideXX.html file if it has actually changed.  This would
#           make uploads faster, when using rsync or ncftpput. [DONE]
#        o Force suffix to be .jpg for thumbnails (sometimes it's JPEG or JPG).
#        o Allow it to work with PNGs (for occasional sketches, scans, etc.)
#        o Remove quote marks from the ALT= text, since that screws things up: ALT=" I said "hello!""
#        o After making all the slides and thumbs, go thru and delete all the old files associated with 
#           images that have been moved, deleted, renamed, etc. The size of these can be many x the size of
#           the main collection.
#        o Make captions_photo2web so that captions are extracted only for the new files. Right now if one image is redone, all the 
#           captions are re-extracted. [DONE]
#        o If photos.css doesn't exist, should copy it from above. Same for header.txt.
#        o captions.txt is not run the first time around. It should be.
#        o In LR, I can put a hard CR or a soft CR into the caption field. If I put a hard CR in (shift-return), then it loads
#          as some sort of funny HTML Euro symbol thingy. I should be able to just do a search & replace for that string here,
#          to cut that out.
#        o Change the formatting so it's in a 3-column CSS window. Put the images themself (in 'all photos' mode) in a div with a fixed 
#          width. The photos are fixed width already (700 pix); make the text fixed also. [DONE -- see New Horizons encounter CSS file]
#        o Eliminate the sections.txt file. Instead, put a tag in the captions ("## On to Nebraska...") so they will auto-adjust. [DONE]

# require "/System/Library/Perl/5.12/ctime.pl";  # Deprecated with Yosemite. Replaced with scalar(localtime())

require "/Users/throop/bin/getopts.pl";   # Process commandline options

$inhibit 	= ".photo2web_inhibit";
$authorline 	= "<!Created by photo2web, Doug Hamilton and Henry Throop, throop at boulder.swri.edu>\n";
$faviconline    = '<link rel="icon" type="image/png" href="$IMAGES/favicon_eggplant.png">';

$cssfile	= "photos.css";
$DO_VERTICAL_CAPTIONS = 0;

&Getopts('vh');	

if ($opt_h) {
  print "photo2web_old help\n";
  print "--------------\n";
  print "  -v    vertical captions\n";

  exit();
}

if ($opt_v) {
  print "Doing vertical captions...\n";
  $DO_VERTICAL_CAPTIONS = 1;
}

if (-e $inhibit){
  die "File $inhibit found -- stopping!\n";
}

if (-e $cssfile){
  $do_css = 1;
  print "Using photos.css\n";

} else {
  $do_css = 0;
}

# First process the files, and copy into 'originals' as needed
# Create the 'originals' directory if it doesn't exist.

if (!(-e 'originals')){ 
  system ("mkdir originals");
  system ("cat /dev/null > originals/.photo2web_inhibit");	# Put a file here so we won't accidentally run it from originals/
};

# Get a list of all the images in the 'originals' directory

open(FILE, "ls -a originals |");
@JPGfilesOriginals = (zeroelem);     # Get all .JPG files to convert
while (<FILE>) {  if (/\.JPG/ || /\.jpg/ || /\.jpeg/) {push(@JPGfilesOriginals, $_);} }

# Get a list of all the images in the input directory
# Copy each of them to the 'originals' directory, renaming where appropriate if one of the files
# is a revised version of an existing file. Rename so as to keep the order (001_, 002_, etc.) unchanged.


$num_copied = 0;

print "Reading new and old file lists\n";

open(FILE, "ls -a |");
@JPGfiles = (zeroelem);     # Get all .JPG files to convert
while (<FILE>) {  
  if (/\.JPG/ || /\.jpg/ || /\.jpeg/) {
    $file = $_; chop $file;
#     print "Image file $file\n";
    if (/^[0-9][0-9][0-9]_/) {
      $file_base = substr($file, 4);		# Get the basename of the file (001_DSC12.jpg -> DSC12.jpg)
#       print "Image file base $file_base\n";
      $copied = 0;
      for ($i=1; $i<=$#JPGfilesOriginals; $i++) {	# Look for any file in originals/ with that same basename
	$file_original = @JPGfilesOriginals[$i]; chop $file_original;
	$file_base_original = substr($file_original, 4);
	if ($file_base eq $file_base_original) { # If it's there, copy new onto old, with old filename
	  $cmd = "mv \"$file\" \"originals/$file_original\"";
	  print $cmd . "\n";
 	  system($cmd);
	  $copied = 1;					# Set flag
#	  $num_copied++;                                # For new extractor, don't set flag

          # Extract the caption for this image, using new python version of extractor

	  $cmd = "captions_photo2web $i";
	  print $cmd;
	  system($cmd);

	} else {
# 	  print "files do not match: $file_base, $file_original\n";
	}
      }
      if (not($copied)) {				# If no match found, then copy the file in directly
	$cmd = "mv \"$file\" \"originals/\"";
#         print $cmd . "\n";
 	system($cmd);
        $copied = 1;
	$num_copied++;
      }
    } else {						# If it's not in form 001_, then just copy it in directly
        $cmd = "mv \"$file\" \"originals/\"";
 	system($cmd);
      }
  }
}

# Re-extract the captions, if necessary.
# We do this if a) there is already a captions.txt, and b) we just copied at least one file above
# 
# Ideally we would do a selective caption extraction, and only extract the one that we need! But for now, the
# captions.txt file is one monolithic file, so it's hard to do just one. In a future revision, I should consider
# making a captions/ folder, with 001.txt, 002.txt, etc, and then it would be very easy to extract just the 
# single necessary caption. This step is definitely a real bottleneck as it is now.

if ((-e "captions.txt") and ($num_copied > 0)) {
  print "Extracting all captions\n";
  system("captions_photo2web");
}

# Now start the main code

open(FILE, "ls -a originals |");
@JPGfiles = (zeroelem);     # Get all .JPG files to convert
while (<FILE>) {  
    if (/\.JPG/ || /\.jpg/ || /\.jpeg/) {
    push(@JPGfiles, $_);
#     print "Added file " . $_;
    }
}

if ($#JPGfiles == 0){
  die("photo2web_old: No photos found.\n");

print "$#JPGfiles Pictures detected\n";
}
close(FILE);

# Produce names for thumbnail images and slides:
# @Slidefile --> slides names
# @medium  --> medium thumbnail
# @intermediate  --> intermediate thumbnail
# @small   --> small thumbnail

for ($i=1;$i<=$#JPGfiles;$i++) {
    chop(@JPGfiles[$i]);
    @small[$i] =  "thumbnails/s@JPGfiles[$i]";
    @intermediate[$i] = "thumbnails/i@JPGfiles[$i]";
    @medium[$i] = "thumbnails/m@JPGfiles[$i]";

    @Slidefile[$i] = "slides/Slide$i.html";

    if ($i< 100) {
        @Slidefile[$i] = "slides/Slide0$i.html";
    }
    if ($i< 10) {
        @Slidefile[$i] = "slides/Slide00$i.html";
    } 

}

if (-e "captions.txt") {
    @captions = &getcaptions();
}

if (-e "../rss.html") {
    @rss = &getrss();
}

if (-e "header.txt") {
    @header = &getheader();
}

if (1) {
    $ContactInfo = &getcontactinfo();
}

@sections = &getsections2();		# Read 'sections' from the captions


if (-e "sections.txt") {		# Read 'sections' from the sections.txt file, which overrides if it exists.
     @sections = &getsections();
}

if ((-e "colors.txt") || (-e "photos.css")) {
    $ColorString = &getcolors();
} 
else {
  print "Using default colors\n";
  $ColorString = 
  "bgcolor=black text=white link=cornflowerblue vlink=lightsalmon alink=lightsalmon";
}

&makeIndex();       # make index.html w/ small thumbnails & links
&makeSinglePageShow();       # make show.html w/ images and captions
if (-e "thumbnails" == 0)  {
    system ("mkdir thumbnails");
}
&mogrifyImages();   # Mogrify the images
if (-e "slides" == 0)  {system ("mkdir slides");}
&makeSlides();      # make Slides from medium thumbnails


sub getcolors {
    local($infile, $ColorString);
    print "Reading Colors\n";
    $infile = "colors.txt";
    open(FILE, $infile);
    $ColorString = <FILE>;
    close(FILE);
    $ColorString;
}

sub getsections {		# Get sections from the sections.txt file.
    local($infile,$i);
    print "Reading Sections\n";
    $infile = "sections.txt";
    open(FILE, "<$infile");
    @sections = ("zeroelement", &ReadInputParagraphs);
    for ($i=1;$i<=$#sections;$i++) { 
        $title = $sections[$i];
    }
    @sections;
}

sub getsections2 {		# Extract sections from tags ("## New Section") in the individual captions. Need not be own line.
    local($infile, $i);
    print "Readings captions.txt to get sections\n";
    $infile = "captions.txt";
    open(FILE, "<$infile");
    @captions_tmp = ("zeroelement", &ReadInputParagraphs); # Read them all
    @sections = ("zeroelement");
    for ($i=1; $i<= $#captions_tmp; $i++) {
    $line = @captions_tmp[$i];
      $pos = index($line, "##");	# Extract everything in that caption after the '##' and call it a section.
      if ($pos >= 0) {
        $name = substr($line, $pos + 2);
        push (@sections, "$i $name");	# Stuff the number and section text into a string, in same format it would have been read from file.
#         print "Section @ $i : $name; full = $line\n";
      }

    }
    @sections;
}

sub getcaptions {
    local($infile,$i);
    $infile = "captions.txt";
    if (-e "originals/captions.txt") {
      $infile = "originals/captions.txt";
    }
    open(FILE, "< $infile");
    @captions = ("zeroelement", &ReadInputParagraphs);
    print "$#captions captions read\n";
    if ($#captions != $#JPGfiles) {
        print "Mismatch! $#captions captions and $#JPGfiles Images!\n";
        for ($i=1;$i<=$#captions;$i++) {
            print "$i - $JPGfiles[$i] - @captions[$i] - \n";
        }
        die "No captions applied to images.\n";
        @captions=();
    }

# If caption has 'jpg' tag in it, then don't print any text.
    for ($i=1;$i<=$#captions;$i++) {
      $_ = @captions[$i];
      if ((/jpg$/) || (/JPG$/) || (/jpeg$/) || (/JPEG$/) || (/png$/) || (/PNG$/)) {
        @captions[$i] = "";
      }

# If caption has a flag ("##") for a section name, delete that text out

      if (/\#\#/) {
        @captions[$i] = substr($_, 0, index($_, "##"));
      }
    }

    close(FILE);
    @captions;
}


# Read the rss.html file

sub getrss {
    local ($infile,$i);
    print "Reading RSS\n";
    $infile = "../rss.html";
    open(FILE, " < $infile");
#     @rss = <FILE>;
    @rss = ("zeroelement", &ReadInputParagraphs);
    close(FILE);
#     print "RSS: " . @rss[0];
    @rss;
}

# Read in the header.txt file.
# The first paragraph is used as the <TITLE> of *all* the slides.
# The second paragraph is used as the header of the main index.html file.

sub getheader {
    local ($infile,$i);
    print "Reading Header\n";
    $infile = "header.txt";
    open(FILE, "< $infile");
    @header = ("zeroelement", &ReadInputParagraphs);
    close(FILE);
    @header;
}

sub getcontactinfo {
    local ($ContactInfo);
    print "Reading contact info\n";
    if (-e "contact_info.txt"){
      $infile = "contact_info.txt";
      open(FILE, $infile);
      $ContactInfo = <FILE>;
      close(FILE);
    } else {
      $ContactInfo = "Henry Throop";
    }

    $ContactInfo;
}

# set Filehandle FILE before calling ReadInputParagraphs
# takes no arguments, returns a list
sub ReadInputParagraphs {
    local($paragraphs,$thefile);
    $/ = "xxxxXXXXxxxx";   # enable whole file mode (no reg. exp.)
#     $* = 1;  # enable multi-line patterns  *** DEPRECATED! Use /m instead.
    $thefile = <FILE>; # read the whole file into a variable
    $thefile =~ s/(\S)\s*\n+\s*\n+\s*(\S)/$1\n\n$2/g;  # white spaces
    @paragraphs = split(/\n\n/,$thefile);
    while (@paragraphs[$#paragraphs] =~ /^\s+$/) {  
        pop(@paragraphs);   # remove blank lines from end of file
    }
    @paragraphs;
}

# Takes a line, and subtracts its first word
sub MinusFirstWord {
    local($in) = @_;
    $in =~ s/^\w+\s+//;
    $in =~ s/\s+$//;
    $in;
}

# Takes a line, and returns only its first word
sub FirstWord {
    local($in) = @_;
    $in =~ s/\s.+$//;
    $in =~ s/\s+//;
    $in;
}

sub makeSinglePageShow {
    local($i,$html,$outfile,$intro,$j);

    $outfile = "show_old.html";

    $photo_index_link = "<p><a href=\"..\"><img src=\"../icons/info.gif\" border=0>Return to list of galleries</a><p>\n";
    if (-e "INHIBIT_PHOTO_INDEX"){
      $photo_index_link = "";
      print "Inhibiting Photo Index link\n";
    }

    open(FILE2, "> $outfile");
    print FILE2  $authorline;
    $headerline = "<HTML>\n <HEAD> <TITLE>" . $header[1] . "</TITLE>\n $faviconline\n </HEAD>\n";
    $headerline =~ s/\$IMAGES/\.\./g;
    print FILE2  $headerline;

    if ($do_css) {
      print FILE2 "<link rel=\"stylesheet\" type=\"text/css\" href=\"photos.css\">\n";
    } else {
      print FILE2  "<body " . $ColorString . ">";
    }

    $header[2] =~ s/\$IMAGES/\.\./g;		# Convert $IMAGES -> ..

    print FILE2 "<div class=header><p>";
    print FILE2 $rss[1];			# Print all of the header lines
    for ($j=2; $j<=$#header; $j++) {
      print FILE2 $header[$j];
    }
    print FILE2 "</div>";
    $intro = "<FONT SIZE=+1><B>\n";
    $intro = "$intro <p><a href=\"index.html\"><img src=\"../icons/RightArrow.gif\" border=0>Thumbnails";
    if (-e "captions.txt") {$intro = "$intro";}
#    if (-e "captions.txt") {$intro = "$intro (no captions)";}
    $intro = "$intro</a><p>\n";
    $intro = "$intro <p><a href=\"@Slidefile[1]\"><img src=\"../icons/RightArrow.gif\" border=0>Slideshow (big images)</a><p>\n";
    $intro = "$intro $photo_index_link";
    $intro = "$intro </B></FONT>\n"; 
    
    if ($DO_VERTICAL_CAPTIONS) {
      $tablestart	= "<table align=middle cellspacing=5 id=photoCellVertical>";
    }
    else {
      $tablestart	= "<table align=middle cellspacing=50 id=photoCellHorizontal>";
    }

    $tableend	= "</table>";

    print FILE2 $intro; 

# Make a list of all of the sections, along with a quick jump-to link to each section

    if ($#sections > 0){
      print FILE2 "<FONT SIZE=+0></B>Jump to section:<br><br>";
      print FILE2 "</FONT></B>";

      for ($j=1; $j<=$#sections; $j++) {
        print FILE2 "&nbsp;&nbsp;&nbsp;<a href=show_old.html\#$j>" . MinusFirstWord(@sections[$j]) . "</a><br>\n";
      }
    }

# Start the main list of images

    print FILE2 $tablestart;
    for ($i=1; $i<=$#JPGfiles; $i++) { 
        
        $html="<tr><td align=middle><a href=@Slidefile[$i]>" .  # was @JPGfiles[$i] to make it display full-res image
	      "<img border=0 src=\"@intermediate[$i]\"></a></td>\n" .
	      "<td cellpadding=10>\n" . 
	      "<div class=caption><p>\n" . @captions[$i] ."\n</p></div>\n</td></tr>\n\n"; 
								# Wrap the caption in a <p> tag for css.
								# But I should really use a named tag for that.

        $html_vertical="<tr><td align=left><a href=@Slidefile[$i]>" .  
	      "<img border=0 src=\"@intermediate[$i]\"></a></td></tr>\n" .
	      "<tr><td cellpadding=0 align=left>\n" . @captions[$i] ."<br><br><br></td></tr>\n\n"; 

        if ($DO_VERTICAL_CAPTIONS) {
	  $html = $html_vertical;
	}

        for ($j=1; $j<=$#sections; $j++) {
	    if ($i eq &FirstWord(@sections[$j])) {
               print FILE2 $tableend;
	       print FILE2 "<a name=" . $j . ">\n";
	       print FILE2  "<hr><h3 align=center>" . MinusFirstWord(@sections[$j]) . "</h3>\n" . $tablestart . "\n";
	    }
	}

	$html =~ s/\$IMAGES/\.\./g;	# Convert $IMAGES -> .. so as to make relative URL

	$_ = @intermediate[$i];

			# If there is a panorama file, stop the tabulated output, do the pan, and restart the table.	
			# If we don't do this, the pan is so fat that the captions for *every single photo* will be pushed
			# far to the right. Breaking the table avoids that.
	if ((/pan/) || (/Pan/)) {
	  print FILE2 $tableend;
	  print FILE2 $tablestart;
	  print FILE2 $html;
	  print FILE2 $tableend;
	  print FILE2 $tablestart;
	} else {
          print FILE2 $html;  
	}
    }
    print FILE2 "</table align=middle>";

    print FILE2 "<hr>" .
      "<FONT SIZE=+1><B> " . $photo_index_link .
      "</B></FONT>" .
      "<hr>" . 
      "<ADDRESS> " . $ContactInfo . "</ADDRESS>" .
      "<p>Last modified " . scalar(localtime()) . "</p>" .
      "</BODY>" .
      "</HTML>";

    close(FILE2);
    print "Updated show_old.html\n";
}


# Make the main index.html page, with thumbnails for each image

sub makeIndex {
    local($i,$html,$outfile,$intro,$j);

    $outfile = "index_old.html";
    open(FILE2, "> $outfile");
    print FILE2  $authorline;
    $headerline = "<HTML>\n <HEAD> <TITLE>" . $header[1] . "</TITLE>\n $faviconline\n </HEAD>\n";
    $headerline =~ s/\$IMAGES/\.\./g;
    print FILE2  $headerline;

    $photo_index_link = "<p><a href=\"..\"><img src=\"../icons/info.gif\" border=0>Return to list of galleries</a><p>\n";
    if (-e "INHIBIT_PHOTO_INDEX"){
      $photo_index_link = "";
    }

    if ($do_css) {
      print FILE2 "<link rel=\"stylesheet\" type=\"text/css\" href=\"photos.css\">\n";
    } else {
      print FILE2  "<body " . $ColorString . ">";
    }



    print FILE2 "<div class=header><p>";
    print FILE2 $rss[1];			# Print all of the header lines
    for ($j=2; $j<=$#header; $j++) {
      print FILE2 $header[$j];
    }
    print FILE2 "</div>";


#     print FILE2 $rss[1];			# Print all of the header lines
#     for ($j=2; $j<=$#header; $j++) {
#       print FILE2 $header[$j];
#     }

    $intro = "<FONT SIZE=+1><B>\n";
    $intro = "$intro <p><a href=show_old.html><img src=\"../icons/RightArrow.gif\" border=0>View all images";
    if (-e "captions.txt") {$intro = "$intro with captions on one page";}
    $intro = "$intro</a><p>\n";
    $intro = "$intro <p><a href=\"@Slidefile[1]\"><img src=\"../icons/RightArrow.gif\" border=0>Slideshow (big images) </a><p>\n";
    $intro = "$intro $photo_index_link";
    $intro = "$intro </B></FONT>\n"; 
    print FILE2 $intro; 
    for ($i=1;$i<=$#JPGfiles;$i++) { 
        $html="<a href=@Slidefile[$i]><img border=0 src =\"@small[$i]\" alt=\"@captions[$i]\" ></a>\n"; 
        $html="<a href=@Slidefile[$i]><img border=0 src =\"@small[$i]\"                       ></a>\n"; 
        for ($j=1;$j<=$#sections;$j++) {
	    if ($i eq &FirstWord(@sections[$j])) {
               print FILE2 "<p><hr>";
	       print FILE2 "<a name=" . $j . ">\n";
	       print FILE2 "<h3 align=center>" . MinusFirstWord(@sections[$j]) . "</h3><p>\n";
	    }
	}
        print FILE2 $html;  
    }
    print FILE2 "<hr>" .
      "<FONT SIZE=+1><B> " . $photo_index_link . 
      "</B></FONT>" .
      "<hr>" . 
      "<ADDRESS>" . $ContactInfo . "</ADDRESS>" .
      "<p>Last modified " . scalar(localtime()) . "</p>" .
      "</BODY>" .
      "</HTML>";

    close(FILE2);
    print "Updated index_old.html\n";
}

sub mogrifyImages {
    local($com);
    $background = " & ";
    for ($i=1;$i<=$#JPGfiles;$i++) {

# We want to skip creating the mogrified image if the target already
# exists, and is newer than the source file.

      $modtime1 = (stat("originals/" . @JPGfiles[$i]))[9];
      $modtime2 = (stat(               @small[$i]))[9];
      $dt = $modtime1 - $modtime2;	# get the time in seconds between 
      					# original and thumbnail
      if (($dt > 0) || (-e @small[$i] == 0)) {
        print "Mogrifying file @JPGfiles[$i] ... ($i of $#JPGfiles)\n";
        $com =  "cp \"originals/@JPGfiles[$i]\" \"@small[$i]\" \n";	# Quote so spaces work
        system($com);
        $com =  "cp \"originals/@JPGfiles[$i]\" \"@intermediate[$i]\" \n";
        system($com);
        $com =  "cp \"originals/@JPGfiles[$i]\" \"@medium[$i]\" \n";
        system($com);
#         $com = "convert -geometry 128x128 \"@small[$i]\" \"@small[$i]\" \n";

# If this is a pan image, them double the size of the thumbnail
# Pans are identified by having 'pan' in the filename somewhere (e.g., "DSC_1234_pan.jpg")

        $_ = @small[$i];
        if ((/pan/) || (/Pan/)) {
          $com = "convert -size 256x256 \"@small[$i]\" -resize 256x256 \"@small[$i]\" $background \n";
	  print "Making 'pan' small\n"
        }
	else {
          $com = "convert -size 256x256 \"@small[$i]\" -resize 256x256 \"@small[$i]\" $background \n";
	}
#         print $com;
        system($com);

# Do the actual Imagemagick resizes.  I don't know with the '>' were used before -- they
# are some sort of constraint.  Perhaps what they do is only allow images to 
# shrink, not get bigger.  But sometimes I want them to get bigger too (eg, ms erin's 
# camping photos, which were tiny, but I wanted to post).

#         $com = "convert -geometry \"500x500\>\" \"@intermediate[$i]\" \"@intermediate[$i]\" \n"; 
        $_ = @small[$i];

# For the 'intermediate', don't make them double the normal size, since this
# messes up page layout. But make them bigger than normal.

        if ((/pan/) || (/Pan/)) {
          $com = "convert -size \"1100x1100\" \"@intermediate[$i]\" -resize \"1100x1100\" \"@intermediate[$i]\" $background \n"; 
	  print "Making 'pan' intermed\n"

	} else {
          $com = "convert -size \"700x700\" \"@intermediate[$i]\" -resize \"700x700\" \"@intermediate[$i]\" $background \n"; 
	}

#         print $com;
        system($com);
#         $com = "convert -geometry \"1000x1000\>\" \"@medium[$i]\" \"@medium[$i]\" \n"; 
#         $com = "convert -size \"1000x1000\>\" \"@medium[$i]\" -resize \"1000x1000\" \"@medium[$i]\" \n"; 
        $_ = @small[$i];
        if ((/pan/) || (/Pan/)) {
          $com = "convert -size \"2000x2000\" \"@medium[$i]\" -resize \"2000x2000\" \"@medium[$i]\" $background \n"; 
	  print "Making 'pan' medium\n"
	} else {
          $com = "convert -size \"1000x1000\" \"@medium[$i]\" -resize \"1000x1000\" \"@medium[$i]\" $background \n"; 
	}
#         print $com;
        system($com);
      }
    }
}


sub makeSlides {
    local($last,$i,$next);
    for ($i=1;$i<=$#JPGfiles;$i++) {    
        if ($i==1) {$last=$#JPGfiles;} else {$last=$i-1;}
        if ($i==$#JPGfiles) {$next=1;} else {$next=$i+1;}
	$slidetmp = "slides/slide_tmp.html";
        open(FILE3, "> $slidetmp");
#         open(FILE3, "> @Slidefile[$i]");
	print FILE3  $authorline;
        $headerline = "<HTML>\n <HEAD> <TITLE>" . $header[1] . "</TITLE>\n $faviconline\n </HEAD>\n";
        $headerline =~ s/\$IMAGES/\.\.\/\.\./g;		# $IMAGES -> ../../
        print FILE3  $headerline;

        if ($do_css){
          print FILE3 "<link rel=\"stylesheet\" type=\"text/css\" href=\"../photos.css\">\n";
	  print FILE3 "<div class=caption>";
        } else {
          print FILE3  "<body " . $ColorString . ">";
        }

# Get the caption text. Convert the string $IMAGES -> ../.. , so as to make a relative URL

        &gethtmlstring($last,$i,$next);
	$htmlstring =~ s/\$IMAGES/\.\.\/\.\./g;
        print FILE3 $htmlstring;		# Output the text of the slides itself
        if ($do_css){
          print FILE3 "</div>";
	}
        close(FILE3);

# Now go and see if the new slide file actually differs from the old one.
# If they are different, copy from tmp to real file.  If not, leave it alone.
# This is so that when we sync to web server, we don't have 100 'modified' files,
# even if only one has a change.  The photo mogrifier uses this same logic -- only update
# the file if we have to.

        if (-e @Slidefile[$i]) {
	  $diff = `diff @Slidefile[$i] $slidetmp`;
# 	print "length diff = " . length($diff) . "\n";
	  if (length($diff) != 0) {
#	    print  "Updating slide @Slidefile[$i]\n";
	    system("cp $slidetmp @Slidefile[$i]");
	  } else {
#	    print "Slide @Slidefile[$i] unchanged\n";
	  }
	} else {
	    print  "Creating slide @Slidefile[$i]\n";
	    system("cp $slidetmp @Slidefile[$i]");
	}
	  
    }
#     system ("rm $slidetmp");
    print "@Slidefile[1] - @Slidefile[$#JPGfiles] Updated\n";
}

sub gethtmlstring {
    local($last,$this,$next)=@_;
    local($lastslide,$thisslide,$nextslide);
    local($LeftArrow, $index,$RightArrow);
    local($icons);
    $lastslide     = "../@Slidefile[$last]";
    $thisimage     = "../thumbnails/m@JPGfiles[$this]"; 
    $nextslide     = "../@Slidefile[$next]";
    $icons 	   = "../../icons";  
    $LeftArrowDim  = "$icons/LeftArrowDim.gif";
    $LeftArrow     = "$icons/LeftArrow.gif";
    $RightArrowDim = "$icons/RightArrowDim.gif";
    $RightArrow    = "$icons/RightArrow.gif";
    $Index         = "$icons/info.gif";

# Now create the html for each slide, with image, caption, etc.  
# Should be done much more beautifully than this.
# I use three special cases, depending on whether this is first, middle, or 
# last slide in the set.  First and last have dimmed arrow icons and no links.
    
    $htmlstring = <<"end_of_html";

Image $this of $#JPGfiles:
<A HREF="$lastslide">
<IMG ALIGN=middle SRC="$LeftArrow" border=0></A>

<A HREF= "../index_old.html"> 
<IMG ALIGN=middle SRC="$Index" border=0></A>

<A HREF="$nextslide">
<IMG ALIGN=middle SRC="$RightArrow" border=0></A> 

<A HREF="../originals/@JPGfiles[$this]">
Full Resolution</A> <P>    

<big>
@captions[$this] <P>

<A HREF="../originals/@JPGfiles[$this]"><IMG border=0 SRC= "$thisimage"></A>

<LINK REL="PREFETCH" HREF="../@Slidefile[$this+1]">
<LINK REL="PREFETCH" HREF="../thumbnails/m@JPGfiles[$this+1]">

<hr>
    
end_of_html

    if ($this==1){
    $htmlstring = <<"end_of_html";

Image $this of $#JPGfiles:

<IMG ALIGN=middle SRC="$LeftArrowDim" border=0>

<A HREF= "../index_old.html"> 
<IMG ALIGN=middle SRC="$Index" border=0></A>

<A HREF="$nextslide">
<IMG ALIGN=middle SRC="$RightArrow" border=0></A> 

<A HREF="../originals/@JPGfiles[$this]">
Full Resolution</A> <P>    

<big>
@captions[$this] <P>

<A HREF="../originals/@JPGfiles[$this]"><IMG border=0 SRC= "$thisimage"></A>
<LINK REF="PREFETCH" HREF="../@Slidefile[$this+1]">

<hr>
    
end_of_html
  }

    if ($this==$#JPGfiles){
    $htmlstring = <<"end_of_html";

Image $this of $#JPGfiles:

<A HREF="$lastslide">
<IMG ALIGN=middle SRC="$LeftArrow" border=0></A>

<A HREF= "../index_old.html"> 
<IMG ALIGN=middle SRC="$Index" border=0></A>

<IMG ALIGN=middle SRC="$RightArrowDim" border=0>

<A HREF="../originals/@JPGfiles[$this]">
Full Resolution</A> <P>    

<big>
@captions[$this] <P>

<A HREF="../originals/@JPGfiles[$this]"><IMG border=0 SRC= "$thisimage"></A>
<hr>
    
end_of_html
  }
}
