# -*- coding: utf-8 -*-
"Standard HBT function library"
# Created on Wed Nov  5 20:59:35 2014

# To use: 'import HBT as hbt'
# Could do 'from HBT import *', but I don't think importing into global space is a good idea.
#
# Then do 'val = hbt.wheremin(np.arange(100))'

# @author: throop

import numpy as np
import astropy
import astropy.modeling
import skimage.transform as skt  # This 'resize' function is more useful than np's
import matplotlib as plt
import cspice
from   astropy.io import fits


# First define some constants. These are available to the outside world, 
# just like math.pi is available as a constant (*not* a function).
# http://stackoverflow.com/questions/5027400/
#  constants-in-python-at-the-root-of-the-module-or-in-a-namespace-inside-the-modu

d2r = np.pi/180.
r2d = 1./d2r

# Now import additional functions into this module
# These are a part of this module now, and accessible via hbt.<function>

from get_fits_info_from_files_lorri import get_fits_info_from_files_lorri
from get_image_nh                   import get_image_nh

# We want to define these as functions, not classes
# They are all general-purpose functions.
# I want to define this all as a class, I guess? Class HBT?
# But when I import it, I want to do this:
#  import HBT as hbt
#  hbt.maxval(arr)    
#########
# Function for wheremin()
#########

##########
# Get a single FITS image header
##########
 
def get_image_header(file, single=False):
    "Return header for a fits file, as a text array."
    "If single set, then return as a single line, not an np array of strings."
    
    hdulist = fits.open(file)
    header = hdulist[0].header
    
    if (single):
#            out = ''
#            for line in header:
#                out += line + '\n'
        return repr(header)
    
    else:
        return header
        
def remove_brightest(arr, frac_max):
    "Clips the brightest values in an array to the level specified" 
    " e.g., frac = 0.95 will clip brightest 5% of pixels)"
    
    clipval_max = np.percentile(arr, frac_max * 100.)
    return np.clip(arr, np.amin(arr), clipval_max)
    
def ln01(arr, offset=0.01):
    "Scale an array logarithmically. Use an offset and ensure that values are positive before scaling."
    
    return np.log(arr - np.amin(arr) + offset)

def scale_image(min, max, polynomial, percentile=False):
    "Applies a pre-calculated scaling to an array"
    pass


def is_array(arg):
    "Return a boolean about whether the passed value is an array, or a scalar."
    "A string is considered *not* to be a scalar."
    
    import collections
    import numpy as np

    return isinstance(arg, (tuple, collections.Sequence, np.ndarray)) and not isinstance(arg, (str, unicode))

def get_range_user(maxrange = 10000):

    "Request a range of input values from the user. No error checking."
    "  *; 1-10; 44   are all valid inputs. If  *  then output is 1 .. maxrange."
    
    inp2 = raw_input("Range of files [e.g. *; 1-10; 44]: ")

    if (inp2.find('-') >= 0):        # Range of files
        min = int((inp2.split('-'))[0])
        max = int((inp2.split('-'))[1])
        range_selected = range(min, max+1)

    elif (inp2.find('*') >= 0):        # All files
        range_selected = range(maxrange)

    else:        # Only a single file
        range_selected = [int(inp2)]
        print 'Range: ' + repr(range_selected)
            
    return range_selected
    
def get_pos_bodies(et, name_bodies, units='radec', wcs=False, 
                     frame='J2000', abcorr='LT', name_observer='New Horizons'):
    
    "Get an array of points for a list of bodies, seen from an observer at the given ET."
    "Result is in RA / Dec radians. If units='pixels', then it is in x y pixels, based on the supplied wcs."
    "name_bodies may be scalar or vector."

    num_bodies = np.size(name_bodies) # Return 1 if scalar, 2 if pair, etc; len() gets confused on strings.
    
    ra  = np.zeros(num_bodies)
    dec = np.zeros(num_bodies)
    
    if num_bodies == 1 and (hbt.is_array(name_bodies) == False):
        arr = np.array([name_bodies])
    else:
        arr = name_bodies
#    quit   
#    i = 0     
    for i,name_body in enumerate(arr):
      st,ltime = cspice.spkezr(name_body, et, frame, abcorr, name_observer)    
      radius,ra[i],dec[i] = cspice.recrad(st[0:3])
    
    if (units == 'pixels'):
        x, y = wcs.wcs_world2pix(ra*r2d, dec*r2d, 0) # Convert to pixels
        return x, y
        
    return ra, dec # Return in radians
      
      
def get_pos_ring(et, num_pts=100, radius = 122000, name_body='Jupiter', units='radec', wcs=False, 
                    frame='J2000', abcorr='LT+S', name_observer='New Horizons'):
    
    "Get an array of points for a ring, at a specified radius, seen from observer at the given ET."
    "Result is in RA / Dec radians. If units='pixels', then it is in x y pixels, based on the supplied wcs."
    
# Now calculate the ring points...

#    radii_ring = np.array([1220000., 129000.])  # Inner and outer radius to plot
    num_pts_ring = 100
    
    ring_lon = np.linspace(0, 2. * np.pi, num_pts_ring)
    ra_ring  = np.zeros(num_pts_ring)
    dec_ring = np.zeros(num_pts_ring)
    
    rot = cspice.pxform('IAU_' + name_body, frame, et) # Get matrix from arg1 to arg2
    
    st,ltime = cspice.spkezr(name_body, et, frame, abcorr, name_observer)
    pos = st[0:3]
#    vel = st[3:6] # velocity, km/sec, of jupiter
    
    for j in range(num_pts_ring):
        xyz = np.zeros(3)
        xyz = np.array((radius * np.cos(ring_lon[j]), radius * np.sin(ring_lon[j]), 0.))
        
        d_j2000_xyz = np.dot(rot,xyz)  # Manual says that this is indeed matrix multiplication
        j2000_xyz = 0 * d_j2000_xyz
        j2000_xyz[0:3] = d_j2000_xyz[0:3] # Looks like this is just copying it

        rho_planet    = pos                     # Position of planet
        rho_ring      = rho_planet + j2000_xyz  # Vector obs-ring
#        dist_ring     = cspice.vnorm(rho_ring)*1000 # Convert to km... CHECK UNITS!
        
        range_out, ra, dec = cspice.recrad(rho_ring) # 'range' is a protected keyword in python!
        
        ra_ring[j] = ra     # save RA, Dec as radians
        dec_ring[j] = dec
   
    if (units == 'pixels'):
        x_ring, y_ring    = wcs.wcs_world2pix(ra_ring*r2d, dec_ring*r2d, 0) # Convert to pixels
        return xring, yring
              
    return ra_ring, dec_ring
    
def fullprint(*args, **kwargs):  # From http://stackoverflow.com/questions/1987694/print-the-full-numpy-array
  "Print a numpy array, without truncating it."
  "This is really slow for large arrays (> 10K)"
  
  from pprint import pprint
  import numpy
  opt = numpy.get_printoptions()
  numpy.set_printoptions(threshold='nan')
  pprint(*args, **kwargs)
  numpy.set_printoptions(**opt)
  
def imsize((size)):
    "Set plot size. Same as using rc, but easier syntax."
    plt.rc('figure', figsize=(size[0], size[1]))
    
def correct_stellab(radec, vel):
    "Corect for stellar aberration."
    "radec is array (n,2) in radians. velocity in km/sec. Both should be in J2K coords."

    radec_abcorr = radec.copy()    
    for i in range(np.shape(radec)[0]):
        pos_i = cspice.radrec(1., radec[i,0], radec[i,1])
        pos_i_abcorr = cspice.stelab(pos_i, vel)
        rang, radec_abcorr[i,0], radec_abcorr[i,1] = cspice.recrad(pos_i_abcorr)

    return radec_abcorr
    
def image_from_list_points(points, shape, diam_kernel):
    "Given an ordered list of xy points, and an output size, creates an image."
    "Useful for creating synthetic star fields."
    
    kernel = dist_center(diam_kernel, invert=True, normalize=True)
    arr = np.zeros(shape)
    dx = shape[0]
    dy = shape[1]
    x = points[:,0]
    y = points[:,1]
    for i in range(len(x)):
        xi = points[i,0]
        yi = points[i,1]
        if (xi >= 0) & (xi + diam_kernel < dx) & (yi >= 0) & (yi + diam_kernel < dy):
            arr[yi:yi+diam_kernel, xi:xi+diam_kernel] = kernel
     
    return arr
    
def set_plot_defaults():
    plt.rc('image', interpolation='None')       # Turn of interpolation for imshow
    plt.rc('image', cmap='Greys')               # Default color table for imshow
    
def wheremin( arr ):
   "Determines the index at which an array has its minimum value"
   index = np.where(arr == np.amin(arr))
   return (np.array([index]).flatten())[0] # Crazy syntax returns either a scalar, or the 0th element of a vector

def commonOverlapNaive(text1, text2):  
  x = min(len(text1), len(text2))  
  while x > 0:  
    if text1[-x:] == text2[:x]:  
      break  
    x -= 1  
  return x 

def dist_center(diam, circle=False, centered=True, invert=False, normalize=False):
    "Returns an array of dimensions diam x diam, with each cell being the distance from the center cell"
    "Works best if diam is an odd integer."    
    xx, yy = np.mgrid[:diam, :diam]
    
    if (centered):
        dist = np.sqrt((xx - (diam-1)/2.) ** 2 + (yy - (diam-1)/2.) ** 2)
    else: 
        dist = np.sqrt((xx)           ** 2 + (yy)           ** 2)
    
    if (invert):
        dist = np.amax(dist) - dist
        
    if (normalize):
        dist = (dist - np.amin(dist)) / np.ptp(dist)
        
    return dist
    
def longest_common_substring(S,T):
    "Given two strings, returns the longest common substring as a set"
    "http://www.bogotobogo.com/python/python_longest_common_substring_lcs_algorithm_generalized_suffix_tree.php"
    m = len(S)
    n = len(T)
    counter = [[0]*(n+1) for x in range(m+1)]
    longest = 0
    lcs_set = set()
    for i in range(m):
        for j in range(n):
            if S[i] == T[j]:
                c = counter[i][j] + 1
                counter[i+1][j+1] = c
                if c > longest:
                    lcs_set = set()
                    longest = c
                    lcs_set.add(S[i-c+1:i+1])
                elif c == longest:
                    lcs_set.add(S[i-c+1:i+1])

    return lcs_set.pop() # Original function returned lcs_set itself. I don't know why -- I just want to extract one element.

def sfit(arr, degree=3, binning=16): # For efficiency, we downsample the input array before doing the fit.
    "Fit polynomial to a 2D array, aka surface."

# For info on resizing, see http://stackoverflow.com/questions/29958670/how-to-use-matlabs-imresize-in-python
    
    shape_small = (np.size(arr,0)/binning, np.size(arr,1)/binning)
    shape_big   = np.shape(arr)

# Create x and y arrays, which we need to pass to the fitting routine

    x_big, y_big = np.mgrid[:shape_big[0], :shape_big[1]]
    x_small = skt.resize(x_big, shape_small, order=1, preserve_range=True)
    y_small = skt.resize(y_big, shape_small, order=1, preserve_range=True)
    
    arr_small = skt.resize(arr, shape_small, order=1, preserve_range=True)
    p_init = astropy.modeling.models.Polynomial2D(degree=degree)

# Define the fitting routine

    fit_p = astropy.modeling.fitting.LevMarLSQFitter()
        
#    with warnings.catch_warnings():
# Ignore model linearity warning from the fitter
#        warnings.simplefilter('ignore')

# Do the fit itself
        
    poly = fit_p(p_init, x_small, y_small, arr_small)

# Take the returned polynomial, and apply it to our x and y axes to get the final surface fit

    surf_big = poly(x_big, y_big)
    
    return surf_big
    
##########
# Function to determine if an entered string is a number. 
# From http://stackoverflow.com/questions/354038/how-do-i-check-if-a-string-is-a-number-in-python
##########

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

