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
import spiceypy as sp
from   astropy.io import fits
import subprocess
import hbt
import warnings
import importlib  # So I can do importlib.reload(module)
from   photutils import DAOStarFinder
import photutils

# First define some constants. These are available to the outside world, 
# just like math.pi is available as a constant (*not* a function).
# http://stackoverflow.com/questions/5027400/
#  constants-in-python-at-the-root-of-the-module-or-in-a-namespace-inside-the-modu

d2r = np.pi/180.
r2d = 1./d2r

r2as = 60. * 60. / d2r # Radians to arcsec
as2r = 1. / r2as       # Arcsec to radians

# Now import additional functions into this module
# These are a part of this module now, and accessible via hbt.<function>
# Note that for the import to work, the directory of these files must be on PYTHONPATH.
# (ie, git/python_hbt/hbt must be on PYTHONPATH). 
# This seems a little silly, but I think it is true. I must be missing something obvious
# about how modules work because that doesn't seem right.  
# It is necessary for both py2 and py3.

from get_fits_info_from_files_lorri import get_fits_info_from_files_lorri
from read_lorri                     import read_lorri
from read_alice                     import read_alice
from met2utc                        import met2utc
from nh_create_straylight_median    import nh_create_straylight_median
from nh_get_straylight_median       import nh_get_straylight_median
from nh_create_straylight_median_filename import nh_create_straylight_median_filename
from nh_jring_process_image         import nh_jring_process_image
from create_backplane               import create_backplane
from calc_offset_points             import calc_offset_points
from navigate_image_stellar         import navigate_image_stellar

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
# Define a range. This is more useful than python's np.arange(), which doesn't allow for logarithmic steps.
# Syntax is identical to HBT's IDL frange() function.
##########
# NB: I could use np.linspace and np.logspace here... in fact, probably better. However, they use slightly different 
# conventions for start and end locations, etc. so it's fine to just do it explicitly here like I do.

#def frange(start, end, linear=True, log=False, *args):
def frange(start, end, *args, **kwargs):

    '''
    Define a range, using either linear or logarithmic spacing. 
    Both the start and end values are used as bins.
    Syntax: frange(start, end, [num_bins], linear=True, log=False)
    '''

# 31-Aug-2016: Changed from arange() to linspace() since frange(0,1,1000) was returning 1001 bins, due to rounding.
    
# First check if number was passed via *args. If not, figure out what it should be.    

    linear = True         # Default is for a linear range
    log    = False

    for key, value in kwargs.items():  # Was 'iteritems' in py3.
        if (key == 'linear'):
            linear = value
        if (key == 'log'):
            log = value
    
    if (len(args) > 0):
        num = args[0]
#        print("num = " + repr(num))
        
    else:
        num = end - start + 1
    
    if (log == False):
        
        arr = np.linspace(start, end, num) # Documentation says linspace() is better than arange()
        
        # If the input arguments are ints, then make sure output array is int also.
        # This is necessary, because unlike IDL, accessing numpy arrays with float indices does not work!
        
        d = np.sum(np.abs(arr - arr.astype(int))) # Do a test to see if values are equivalent to an int.
        
        if (d == 0):
            arr = arr.astype(int)

        return arr
        
    if (log):
        arr = start * ((end/(start * 1.))**(1./(num-1.))) ** np.array(range(num))
        return np.array(arr)

#==============================================================================
# Set font size for xlabel, title, legend, etc.
#==============================================================================
    
def set_fontsize(size=15):
    
    font = {'family' : 'sans-serif',
            'weight' : 'normal',
            'size'   : size}

    plt.rc('font', **font)
        
##########
# Get a single FITS image header
##########
 
def get_image_header(file, single=False):
    '''
    Return header for a fits file, as a text array.
    If single set, then return as a single line, not an np array of strings.
    '''
    
    hdulist = fits.open(file)
    header = hdulist[0].header
    
    if (single):
#            out = ''
#            for line in header:
#                out += line + '\n'
        return repr(header)
    
    else:
        return header

##########
# Merge FITS image and header info from two files
##########
        
def merge_fits_header(file_image, file_header, file_out):
    '''
    Merge the image data ('PRIMARY'.data) from one FITS file, 
    with the header data ('PRIMARY'.header) from another.
    '''
    
    h_image = fits.open(file_image)
    h_header = fits.open(file_header)
    
    h_image['PRIMARY'].header = h_header['PRIMARY'].header
    h_image.writeto(file_out)
    print("Wrote new file to " + file_out)

##########
# Convert from increment to cumulative
##########

def incr2cum(arr, DO_REVERSE=False):

    '''
    Convert from incremental to cumulative.
    Works for 1D array only
    
    This does the same as np.cumsum, but it allows for the DO_REVERSE flag, which that function doesn't.
    
    '''
    
    if DO_REVERSE:
        return (np.cumsum(arr[::-1]))[::-1]
    else:
        return np.cumsum(arr)
    
##########
# Create a power-law distribution
##########

def powerdist(r, mass, q, rho, DO_NORMALIZE = False):

    """
    Assumes radius in r, and creates # in range [r .. r+dr] = c r^-q dr
    bins are expected to be logarithmic.
    /DO_NORMALIZE: every bin has at least one particle
    in it.  the upper size cutoff may be off by 
    a few bins, but mass is conserved.
    """
    
    import math
 
    ratio = r[1]/r[0]

    dr = r * (ratio-1)    # width of bins
    n = r**(-q) * dr
    mout = 4/3 * math.pi * r**3 * n * rho
    n = n * mass / np.sum(mout)
                         # normalize mass
   
    if DO_NORMALIZE:
        lastbin = (np.where (n < 1))(0)
        if (lastbin != -1):
            n[lastbin:] = 0
            mout = 4/3. * math.pi * r**3 * n * rho
            n = n * mass / np.sum(mout)
     
    return n

##########
# Write a string to the Mac clipboard
##########
# http://stackoverflow.com/questions/1825692/can-python-send-text-to-the-mac-clipboard

def write_to_clipboard(str):
    """
    Write an arbitrary string to the Mac clipboard.
    """
    
    process = subprocess.Popen(
        'pbcopy', env={'LANG': 'en_US.UTF-8'}, stdin=subprocess.PIPE)
    process.communicate(str.encode('utf-8'))
        
def remove_brightest(arr, frac_max, symmetric=False):
    """
    Clips the brightest values in an array to the level specified; e.g., frac = 0.95 will clip brightest 5% of pixels).
    If 'symmetric' set, will also clip same fraction from bottom.				
    """

    clipval_max = np.percentile(arr, frac_max * 100.)
    
    if (symmetric):
        clipval_min = np.percentile(arr, (1-frac_max) * 100)
    else:
        clipval_min = np.amin(arr)
								
    return np.clip(arr, clipval_min, clipval_max)

def remove_extrema(arr, frac_max=0.95, symmetric=True):
    """
    Clips the brightest and darkest values in an array.				
    """
								
    return hbt.remove_brightest(arr, frac_max, symmetric)
    
    
def ln01(arr, offset=0.01):
    """
    Scale an array logarithmically. 
    Use an offset and ensure that values are positive before scaling.
    """
    ## I think it is better to use astropy's scaling than this!
    
    return np.log(arr - np.amin(arr) + offset)

def y2bin(val, bins, vals):
    """
    Returns the integer bin number corresponding to the closest y value to a given y.  
    bins: monotonic x value
    y   : monotonic y values
    val : y value for which we want to look up the bin number. Can be scalar or vector.
    
    Result: Same as input. Converted to an integer.
    """
    
    # This routine is a parallel to the one I had in IDL: sx2bin2 ('spline, x-to-bin, v2'). That 
    # used splines only because it had a good interface, not because it needed them computationally.
    
    return (np.rint(np.interp(val, vals, bins))).astype(int)

def x2bin(val, bins):
    """
    Returns the index of a bin, given its value.
    Note that I have x2bin and y2bin, and they are different.
    """
    
    return hbt.y2bin(val, range(np.size(bins)), bins)

def is_array(arg):
    """
    Return a boolean about whether the passed value is an array, or a scalar.
    A string is considered *not* to be a scalar.
    """
    
    import collections
    import numpy as np

    return isinstance(arg, (tuple, collections.Sequence, np.ndarray)) and not isinstance(arg, (str, unicode))

def sizex(arr):
    """
    Return x size of an array
    """
    return(arr.shape[0])

def sizey(arr):
    """
    Return y size of an array. No error handling.
    """
    return(arr.shape[1])

def get_range_user(maxrange = 10000):

    """
    Request a range of input values from the user. No error checking.
    "  *; 1-10; 44   are all valid inputs. If  *  then output is 1 .. maxrange.
    """
    
    inp2 = raw_input("Range of files [e.g. *; 1-10; 44]: ")

    if (inp2.find('-') >= 0):        # Range of files
        min = int((inp2.split('-'))[0])
        max = int((inp2.split('-'))[1])
        range_selected = range(min, max+1)

    elif (inp2.find('*') >= 0):        # All files
        range_selected = range(maxrange)

    else:        # Only a single file
        range_selected = [int(inp2)]
        print('Range: ' + repr(range_selected))
            
    return range_selected

def mm(arr):
    """
    Return the min and max of an array, in a tuple.
    """
    return (np.nanmin(arr), np.nanmax(arr))
    
def get_pos_bodies(et, name_bodies, units='radec', wcs=False, 
                     frame='J2000', abcorr='LT+S', name_observer='New Horizons', dt=0):
    
    """
    Get an array of points for a list of bodies, seen from an observer at the given ET.
    Result is in RA / Dec radians. If units='pixels', then it is in x y pixels, based on the supplied wcs.
    name_bodies may be scalar or vector.
    
    units = 'pixels', 'degrees', 'radians', etc.
    
    dt: An offset in time, applied not to ET, but to time-of-flight uncertainty in s/c location.
    """

    num_bodies = np.size(name_bodies) # Return 1 if scalar, 2 if pair, etc; len() gets confused on strings.
    
    ra  = np.zeros(num_bodies)
    dec = np.zeros(num_bodies)
    
    if num_bodies == 1 and (hbt.is_array(name_bodies) == False):
        arr = np.array([name_bodies])
    else:
        arr = name_bodies
    
    for i,name_body in enumerate(arr):

# If no time-of-flight offset supplied, use a simple calculation
        
        if (dt == 0):             
            st, ltime = sp.spkezr(name_body, et, frame, abcorr, name_observer)

# If time-of-flight offset supplied, then we need to do a more detailed calculation.
# This below is OK, but it ignores aberration entirely (ie, not even LT)

        else:    
            st_sun_obs,ltime  = sp.spkezr(name_observer, et+dt, frame, 'none', 'Sun')  
            st_sun_targ,ltime = sp.spkezr(name_body,     et,    frame, 'none', 'Sun')      
            st = -st_sun_obs + st_sun_targ
      
        radius,ra[i],dec[i] = sp.recrad(st[0:3])

    if (units == 'radians') or (units == 'rad'):
        return ra, dec
    
    if (units == 'degrees') or (units == 'deg'):
        return ra*hbt.r2d, dec*hbt.r2d
        
    if (units == 'pixels'):
        x, y = wcs.wcs_world2pix(ra*r2d, dec*r2d, 0) # Convert to pixels
        return x, y

    else:    
        return ra, dec # Return in radians
      

##########
# Normalize two images using linear regression
##########

def normalize_images(arr1, arr2, DO_HISTOGRAM=False):
    """Performs linear regression on two images to try to match them.
     Returns fit parameter r: for best fit, use arr2 * r[0] + r[1]
     Goal is to set arr1 to the level of arr2"""
   
    import matplotlib.pyplot as plt
    from   scipy.stats import linregress
    
#       arr1_filter = hbt.remove_brightest(arr1, frac) # Rem
#       arr2_filter = hbt.remove_brightest(arr2, frac)
    r = linregress(arr1.flatten(), arr2.flatten())

    stretch_percent = 90    
    stretch = astropy.visualization.PercentileInterval(stretch_percent) # PI(90) scales to 5th..95th %ile.     

    m = r[0] # Multiplier = slope
    b = r[1] # Offset = intercept

    arr1_norm = arr1 * m + b

    DO_DIAGNOSTIC = False
    
    if DO_DIAGNOSTIC:
        plt.imshow(stretch(arr1))
        plt.title('Arr1 = Stray')
        plt.show()
        
        plt.imshow(stretch(arr1_norm))
        plt.title('Arr1_norm = Stray Norm')
        plt.show()
        
        plt.imshow(stretch(arr2))
        plt.title('Arr2 = Data')
        plt.show()
    
    DO_HISTOGRAM = False
    
    if (DO_HISTOGRAM):
        
        nbins = 20
        
        range1 = (np.amin(arr1), np.amax(arr1))
        (h1, bins1) = np.histogram(arr1, range=range1, bins = nbins)
        
        range2 = (np.amin(arr2), np.amax(arr2))
        (h2, bins2) = np.histogram(arr1, range=range2, bins=nbins)

        range1_norm = (np.amin(arr1_norm), np.amax(arr1_norm))
        (h1_norm, bins1_norm) = np.histogram(arr1_norm, range=range1_norm, bins=nbins)
        
        plt.plot(bins1[0:-1], h1, label = 'Arr1', color = 'blue')
        plt.plot(bins1_norm[0:-1], h1_norm, label = 'Arr1_norm', color = 'lightblue')
        plt.plot(bins2[0:-1], h2, label = 'Arr2', color = 'red')
        
        print("Arr1:      Range = {}".format(range1))
        print("Arr1_norm: Range = {}".format(range1_norm))
        print("Arr2:      Range = {}".format(range2))

        plt.legend()
        plt.yscale('log')
        
        plt.show()
        
    return (arr1_norm, (m,b))

#==============================================================================
# Calculate position of ring points
#==============================================================================
      
def get_pos_ring(et, num_pts=100, radius = 122000, name_body='Jupiter', units='radec', wcs=False, 
                    frame='J2000', abcorr='LT+S', name_observer='New Horizons'):
    
    """
    Get an array of points for a ring, at a specified radius, seen from observer at the given ET.
    Result is in RA / Dec radians. If units='pixels', then it is in x y pixels, based on the supplied wcs.
    """
    
# Now calculate the ring points...
    
    ring_lon = np.linspace(0, 2. * np.pi, num_pts)
    ra_ring  = np.zeros(num_pts)
    dec_ring = np.zeros(num_pts)
    
    rot = sp.pxform('IAU_' + name_body, frame, et) # Get matrix from arg1 to arg2
    
    st,ltime = sp.spkezr(name_body, et, frame, abcorr, name_observer)
    pos = st[0:3]
#    vel = st[3:6] # velocity, km/sec, of jupiter
    
    for j in range(num_pts):
        xyz = np.zeros(3)
        xyz = np.array((radius * np.cos(ring_lon[j]), radius * np.sin(ring_lon[j]), 0.))
        
        d_j2000_xyz = np.dot(rot,xyz)  # Manual says that this is indeed matrix multiplication
        j2000_xyz = 0 * d_j2000_xyz
        j2000_xyz[0:3] = d_j2000_xyz[0:3] # Looks like this is just copying it

        rho_planet    = pos                     # Position of planet
        rho_ring      = rho_planet + j2000_xyz  # Vector obs-ring
#        dist_ring     = sp.vnorm(rho_ring)*1000 # Convert to km... CHECK UNITS!
        
        range_out, ra, dec = sp.recrad(rho_ring) # 'range' is a protected keyword in python!
        
        ra_ring[j] = ra     # save RA, Dec as radians
        dec_ring[j] = dec
   
    if (units == 'pixels'):
        x_ring, y_ring    = wcs.wcs_world2pix(ra_ring*r2d, dec_ring*r2d, 0) # Convert to pixels
        return x_ring, y_ring
              
    return ra_ring, dec_ring
    
def fullprint(*args, **kwargs):  # From http://stackoverflow.com/questions/1987694/print-the-full-numpy-array
  """
  Print a numpy array, without truncating it.
  This is really slow for large arrays (> 10K)
  """
  
  from pprint import pprint
  import numpy
  opt = numpy.get_printoptions()
  numpy.set_printoptions(threshold='nan')
  pprint(*args, **kwargs)
  numpy.set_printoptions(**opt)

def trunc(f, n):
    '''Truncates/pads a float f to n decimal places (not sig figs!) without rounding.'''
    '''Returns string.'''
    # From http://stackoverflow.com/questions/783897/truncating-floats-in-python

    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d+'0'*n)[:n]])

def reprfix(arr):
    ''' This does the same as repr(arr) when passed a numpy arr. However,
    as of numpy 2-Oct-2016, repr(arr) ignores the value of np.set_printoptions(threshold),
    and thus array repr have '...' in them *always*, which is not what I want.
    '''
    
    out = 'array(' + repr(list(arr)) + ')'

    return out
     
def figsize(size=None): # Was imsize(), but I think this is better
    """
    Set plot size to tuple (horizontal, vertical). Same as using rc, but easier syntax.
    """
    
    size_default = (10,6)   # If empty argument, then reset back to default
    
    if (size==None):
        size_out = size_default
    else:
        size_out = (size[0], size[1])
        
    plt.rc('figure', figsize=size_out)
    
def correct_stellab(radec, vel):
    """
    Corect for stellar aberration.
    radec is array (n,2) in radians. velocity in km/sec. Both should be in J2K coords.
    """

    radec_abcorr = radec.copy()    
    for i in range(np.shape(radec)[0]):
        pos_i = sp.radrec(1., radec[i,0], radec[i,1])
        pos_i_abcorr = sp.stelab(pos_i, vel)
        rang, radec_abcorr[i,0], radec_abcorr[i,1] = sp.recrad(pos_i_abcorr)

    return radec_abcorr
    
def image_from_list_points(points, shape, diam_kernel, do_binary=False):  # Shape is (num_rows, num_cols)
                                                
    """
    Given an ordered list of xy points, and an output size, creates an image.
    Useful for creating synthetic star fields. Each point is [row, column] = [y, x]
    
    do_binary: If set, output a boolean mask, rather than a floating point image
    """
    
    # Get the kernel function. Properly mask it so it is a boolean circle.

    if (do_binary): # Create a masked kernel if requested    
        kernel = hbt.dist_center(diam_kernel)
        kernel = (kernel < diam_kernel / 2)

    else:
        kernel = hbt.dist_center(diam_kernel, invert=True, normalize=True)

    # Set up the output array
    
    arr = np.zeros(shape)
    dx = shape[1]
    dy = shape[0]

    x = points[:,1]
    y = points[:,0]

    # Loop over every point
    
    for i in range(len(x)):

        xi = int(round(points[i,1]))
        yi = int(round(points[i,0]))

        if (xi >= 0) & (xi + diam_kernel < dx) & (yi >= 0) & (yi + diam_kernel < dy):
            arr[yi:yi+diam_kernel, xi:xi+diam_kernel] = \
              np.maximum(arr[yi:yi+diam_kernel, xi:xi+diam_kernel], kernel)
              # If we overlap, don't add pixel value -- just take the max of the pair
              
    return arr
    
def set_plot_defaults(cmap='Greys'):
    """ 
    Set default values for matplotlib
    """
    plt.rc('image', interpolation='None')       # Turn of interpolation for imshow
    plt.rc('image', cmap=cmap)               # Default color table for imshow
    
def wheremin( arr ):
   """
   Determines the index at which an array has its minimum value
   """
   
   index = np.where(arr == np.amin(arr))
   return (np.array([index]).flatten())[0] # Crazy syntax returns either a scalar, or the 0th element of a vector

def wheremax( arr ):
   """
   Determines the index at which an array has its max value
   """
   
   index = np.where(arr == np.amax(arr))
   return (np.array([index]).flatten())[0] # Crazy syntax returns either a scalar, or the 0th element of a vector
   
def commonOverlapNaive(text1, text2):  
  x = min(len(text1), len(text2))  
  while x > 0:  
    if text1[-x:] == text2[:x]:  
      break  
    x -= 1  
  return x 

def dist_center(diam, centered=True, invert=False, normalize=False):
    """
    Returns an array of dimensions diam x diam, with each cell being the distance from the center cell.
    Works best if diam is an odd integer.
    Similar to IDL's dist(), although this function applies the shift(shift(dist())).
    If normalize is passed, 
    """
    
    # Create the grid
    
    xx, yy = np.mgrid[:diam, :diam]
    
         # What is this syntax all about? That is weird.
         # A: mgrid is a generator. np.meshgrid is the normal function version.
                                       
    if (centered):
        dist = np.sqrt((xx - (diam-1)/2.) ** 2 + (yy - (diam-1)/2.) ** 2)
    else: 
        dist = np.sqrt((xx)           ** 2 + (yy)           ** 2)
    
    if (invert):
        dist = np.amax(dist) - dist
        
    if (normalize):
        dist = (dist - np.amin(dist)) / np.ptp(dist)
        
    return dist

def smooth_boxcar(ydata, binning):
   """ Do a boxcar smoothing. Array returned has same length as input array.
   """
   
# In theory this works basically the same as astropy.convolution.kernels.convolve with Box1DKernel, 
# but that kept giving me python error, so better to bring it in house.
   
#   ydata_ = ydata.copy()
   
# Add some padding on the edge. Pad with the single edge value
   
   ydata_lhs = np.zeros(binning) + ydata[0]
   ydata_rhs = np.zeros(binning) + ydata[-1]
   ydata_padded = np.concatenate((ydata_lhs, ydata, ydata_rhs))
   
   smoothed_padded = np.convolve(ydata_padded, 1./binning + np.zeros(binning), mode = 'same')
   smoothed = smoothed_padded[binning:-binning]
   
   return smoothed
    
def longest_common_substring(S,T):
    """
    Given two strings, returns the longest common substring as a set.
    """
#    http://www.bogotobogo.com/python/python_longest_common_substring_lcs_algorithm_generalized_suffix_tree.php"
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

    return lcs_set.pop() # Original function returned lcs_set itself. I don't know why -- 
                         # I just want to extract one element.

#==============================================================================
# POLYFIT
#==============================================================================

def polyfit(arr, degree=3):
    
    """
    Fit polynomial to a 1D curve. Returns the value of the value of the curve itself.
    Y values are not specified, but are assumed to be monotonic and evenly spaced.
    """
    
    x = range(np.size(arr))
    
    coeffs_out = np.polyfit(x, arr, degree)
 
    func = np.poly1d(coeffs_out)
    
    return func(x)
    
#==============================================================================
# REMOVE_POLYFIT
#==============================================================================

def remove_polyfit(arr, **kwargs):
    """ Return a function with a polynomial fit removed """
    
    return arr - hbt.polyfit(arr, **kwargs)

#==============================================================================
# SFIT
#==============================================================================
    
def sfit(arr, degree=3, binning=16): # For efficiency, we downsample the input array before doing the fit.
                                     # This binning is OK for FITS files, but should have better error checking!

    import skimage.transform as skt  # This 'resize' function is more useful than np's
                                     
    """
    Fit polynomial to a 2D array, aka surface. Like IDL sfit.
    """

# For info on resizing, see http://stackoverflow.com/questions/29958670/how-to-use-matlabs-imresize-in-python
    
    shape_small = (np.size(arr,0)/binning, np.size(arr,1)/binning)
    shape_big   = np.shape(arr)

    mode        = 'constant' # How to handle edges, in the case of non-integer leftovers. 'constant' or 'reflect'
    
# Create x and y arrays, which we need to pass to the fitting routine

    x_big, y_big = np.mgrid[:shape_big[0], :shape_big[1]]
    x_small = skt.resize(x_big, shape_small, order=1, preserve_range=True, mode=mode)
    y_small = skt.resize(y_big, shape_small, order=1, preserve_range=True, mode=mode)
    
    arr_small = skt.resize(arr, shape_small, order=1, preserve_range=True, mode=mode) # Properly preserves NaN
    p_init = astropy.modeling.models.Polynomial2D(degree=int(degree))

# Define the fitting routine

    fit_p = astropy.modeling.fitting.LevMarLSQFitter()

# astropy insists on warning me to use a linear filtter if my coeffs are linear. This is stupid. Turn off that warning.

    with warnings.catch_warnings():        
        warnings.simplefilter('ignore')

# Extract all of the non-NaN pixels. This is because fit_p will just return zeros if the array 
# has NaN's in it.
        
        is_good = (np.isnan(arr_small) == False)

# Do the fit itself
        
        poly = fit_p(p_init, x_small[is_good], y_small[is_good], arr_small[is_good])

# Take the returned polynomial, and apply it to our x and y axes to get the final surface fit

    surf_big = poly(x_big, y_big)
    
    return surf_big

##########
# REMOVE_SFIT
##########

def remove_sfit(arr, degree=3):
    """
    Return an array with a polynomial fit removed. Same as sfit, but better for inline usage.
    """
    
    return arr - hbt.sfit(arr, degree=degree)
    
def butter2d_lp(shape, f, n, pxd=1):
    """Designs an n-th order lowpass 2D Butterworth filter with cutoff
   frequency f. pxd defines the number of pixels per unit of frequency (e.g.,
   degrees of visual angle)."""

    # See image_filtering_examples.py, and http://www.srmathias.com/image-filtering/

    pxd = float(pxd)
    rows, cols = shape
    x = np.linspace(-0.5, 0.5, cols)  * cols / pxd
    y = np.linspace(-0.5, 0.5, rows)  * rows / pxd
    radius = np.sqrt((x**2)[np.newaxis] + (y**2)[:, np.newaxis])
    filt = 1 / (1.0 + (radius / f)**(2*n))
    return filt
    
def ffit(arr, f=0.1, n=2, pxd=43):
    """
    Applies an FFT filter to fit an image. Returns a low-order fit to the image, similar to sfit.
    The default frequency / cutoff parameters are tuned by hand to fit NH J-ring light-scattering
    data, and are very roughly comparable to sfit(5) under some circumstances.
    """

    # See image_filtering_examples.py, and http://www.srmathias.com/image-filtering/

    fft_orig = np.fft.fftshift(np.fft.fft2(arr))
    recon_image = np.abs(np.fft.ifft2(np.fft.ifftshift(arr)))
 
    filt = butter2d_lp(arr.shape, f, n, pxd=43)
    fft_new = fft_orig * filt
    image_fft = np.abs(np.fft.ifft2(np.fft.ifftshift(fft_new)))
    
    return image_fft
    
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

##########
# Find stars in an image
##########
        
def find_stars(im, num=-1, do_flux=False, sigma=3.0, iters=5, fwhm=5.0, threshold = 9):
    """
    Locate stars in an image array, using DAOphot. 
    Returns N x 2 array with xy positions (ie, column, row). No magnitudes.
    Each star has position [row, column] = [y, x].
    
    Optional: 'num' indicates max number of stars to return, sorted by brightness.
    
    Optional: if do_flux set, then will return a third column, with fluxes.
    
    """

    from   astropy.stats import sigma_clipped_stats
#    from   photutils import daofind # Deprecated

    mean, median, std = sigma_clipped_stats(im, sigma=sigma, iters=iters)
    
    find_func = photutils.DAOStarFinder(threshold, fwhm)
    sources = find_func(im - median)
    
#    sources = daofind(im - median, fwhm=fwhm, threshold=threshold) # Deprecated
    
    sources.sort('flux')  # Sort in-place

    # If the 'num' value is passed, sort, and return only those brightest ones
    
    if (num > 0):  
        index_start = -num
    else:
        index_start = 0
        
    x_phot = sources['xcentroid'][index_start:]
    y_phot = sources['ycentroid'][index_start:]
    flux   = sources['flux'][index_start:]
    
    if do_flux:
        points_phot = np.transpose((y_phot, x_phot, flux)) # Create an array N x 3
    else:     
        points_phot = np.transpose((y_phot, x_phot)) # Create an array N x 2

    return points_phot


#==============================================================================
# Convert from factor, to magnitude
#==============================================================================

# Duplicate IDL hbtlib fac2mag()

def fac2mag(fac):

    import numpy as np

    return np.log( 1/(1.*fac))/np.log(100**0.2)    

#==============================================================================
# Convert from magnitude, factor
#==============================================================================

# Duplicate IDL hbtlib mag2fac()

def mag2fac(mag): 

    import numpy as np
    
    return np.exp(-mag * np.log(100.**0.2))
   
#==============================================================================
# Do a simple cosmic ray rejection
#==============================================================================

def decosmic(im, sigma=3):
    """ Return a cleaned version of an image. Single pixels well above the sigma*stdev
        are replaced with the image median value.
    """
# NB: I should offer to return a mask. And I should offer to replace with something other than the median.
# Logic here is roughly based on https://python4astronomers.github.io/core/numpy_scipy.html

# The bottleneck for this function is the medfilt, which applies a median filter to a given area
# given a kernel width. However, I found that medfilt2d() is 4x as fast as medfilt(): 0.3 vs. 1.3 sec 
# for LORRI 1024x1024.

    import scipy.signal
    import math
        
    WASNAN = -999999
    
    med = np.nanmedian(im)               # Take median, ignoring NaN values
    std = np.nanstd(im)
    im_sm = scipy.signal.medfilt2d(im, 5)  # Smooth the image. Properly handles NaN.

# We need to remove any values which are NaN in either the original input, or the smoothed version of that.

    indices_nan_im    = np.isnan(im)       # Posit 
    indices_nan_im_sm = np.isnan(im_sm)

# Convert the NaNs to something else, to let the math work
    
    im[indices_nan_im]       = WASNAN 
    im_sm[indices_nan_im_sm] = WASNAN

# Do the math. There is not an easy NaN-compatible version of '>' which I found.
    
    is_bad = (im - im_sm) > (std*sigma)  # Flag pixels which drop in value a lot when smoothed
    im_fix = im.copy()                   # Create output array
    im_fix[is_bad] = med                 # Replace pixels

# Any pixels that were NaN before, turn them back into NaN

    im_fix[indices_nan_im]    = math.nan
    im_fix[indices_nan_im_sm] = math.nan
    
    return im_fix
    
#==============================================================================
# Function to get the translation between two sets of points
# This is my brute-force shift-and-add image translation searcher.
# It works perfectly, but is super slow.
# Usually ird.translation() is better. [NB: Need to normalize images properly before ird.translation()!]
#==============================================================================

def get_translation_images_bruteforce(im_a, im_b):
    """
    Return the shift (dy, dx) between a pair of images. Slow.
    """        
    shift_max = 50 # Shift halfwidth
    
    range_dy = np.arange(-shift_max, shift_max+1)  # Actually goes from -N .. +(N-1)
    range_dx = np.arange(-shift_max, shift_max+1)
        
    sum = np.zeros((hbt.sizex(range_dx), hbt.sizex(range_dy))) # Create the output array. 
                                         # Each element here is the sum of (dist from star N to closest shifted star)
    
    for j,dx in enumerate(range_dx):
        for k,dy in enumerate(range_dy):
            sum[j,k] = np.sum(np.logical_and( im_a, np.roll(np.roll(im_b,dx,0),dy,1)) )
        print("j = {}, dx = {}".format(j, dx))
        
    pos = np.where(sum == np.amax(sum)) # Assumes a single valued minimum
    
    dy_out = pos[1][0] - shift_max
    dx_out = pos[0][0] - shift_max
        
    return (np.array((dy_out, dx_out)), sum)

#==============================================================================
# Function to remove some vertical striping from LORRI images.
#==============================================================================

def lorri_destripe(im):
    '''
    This function removes a vertical banding from some LORRI images.
    Banding is caused by a bias offset between odd and even colums.
    This is known as 'jail bars' by Hal Weaver.
    '''
    
    colmean      = np.mean(im, 0)    
    colmean_even = colmean[0::2]
    colmean_odd  = colmean[1::2]
    
# Calculate the bias, as a single scalar value to be subtracted from each even column
    
    bias_even    = np.mean(colmean_even - colmean_odd)

# Create a 2D array to be subtracted
# This is lengthier than it should be. I can probably do something like 
#     arr[0::2] -= bias
# but that is for rows and not columns, and I couldn't figure out the proper syntax!

    im_out         = im.copy()
    arr_bias       = np.zeros(np.shape(im))
    arr_bias[0::2] = bias_even
    arr_bias       = np.transpose(arr_bias)
        
    im_out -= arr_bias
          
    return(im_out)
     