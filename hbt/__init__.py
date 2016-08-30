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
import subprocess
from   scipy.stats import linregress
import hbt

# First define some constants. These are available to the outside world, 
# just like math.pi is available as a constant (*not* a function).
# http://stackoverflow.com/questions/5027400/
#  constants-in-python-at-the-root-of-the-module-or-in-a-namespace-inside-the-modu

d2r = np.pi/180.
r2d = 1./d2r

# Now import additional functions into this module
# These are a part of this module now, and accessible via hbt.<function>

from get_fits_info_from_files_lorri import get_fits_info_from_files_lorri
from read_lorri                     import read_lorri
from read_alice                     import read_alice
from met2utc                        import met2utc
from nh_create_straylight_median    import nh_create_straylight_median
from nh_get_straylight_median       import nh_get_straylight_median
from nh_create_straylight_median_filename import nh_create_straylight_median_filename

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
    
# First check if number was passed via *args. If not, figure out what it should be.    

    linear = True         # Default is for a linear range
    log    = False

    for key, value in kwargs.iteritems():
        if (key == 'linear'):
            linear = value
        if (key == 'log'):
            log = value
    
    if (len(args) > 0):
        num = args[0]
        print "num = " + repr(num)
        
    else:
        num = end - start + 1
    
    if (log == False):
        step = (end - start)/((num-1) * 1.)
        out = np.arange(start, end+step, step)
#        out = np.array(range(num))/(num - 1.) * (end-start) + start

        return out
        
    if (log):
        out = start * ((end/(start * 1.))**(1./(num-1.))) ** np.array(range(num))
        return np.array(out)

        
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
    
def ln01(arr, offset=0.01):
    """
    Scale an array logarithmically. 
    Use an offset and ensure that values are positive before scaling.
    """
    
    return np.log(arr - np.amin(arr) + offset)

def scale_image(min, max, polynomial, percentile=False):
    """
    Applies a pre-calculated scaling to an array. NOT WORKING.
    """
    pass


def is_array(arg):
    """
    Return a boolean about whether the passed value is an array, or a scalar.
    A string is considered *not* to be a scalar.
    """
    
    import collections
    import numpy as np

    return isinstance(arg, (tuple, collections.Sequence, np.ndarray)) and not isinstance(arg, (str, unicode))

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
        print 'Range: ' + repr(range_selected)
            
    return range_selected

def mm(arr):
    """
    Return the min and max of an array, in a tuple.
    """
    return (np.min(arr), np.max(arr))
    
def get_pos_bodies(et, name_bodies, units='radec', wcs=False, 
                     frame='J2000', abcorr='LT', name_observer='New Horizons'):
    
    """
    Get an array of points for a list of bodies, seen from an observer at the given ET.
    Result is in RA / Dec radians. If units='pixels', then it is in x y pixels, based on the supplied wcs.
    name_bodies may be scalar or vector.
    """

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
      

##########
# Normalize two images using linear regression
##########

def normalize_images(arr1, arr2):
    "Performs linear regression on two images to try to match them."
    "Returns fit parameter r: for best fit, use arr2 * r[0] + r[1]"
    "Goal is to set arr1 to the level of arr2"
    
#       arr1_filter = hbt.remove_brightest(arr1, frac) # Rem
#       arr2_filter = hbt.remove_brightest(arr2, frac)
    r = linregress(arr1.flatten(), arr2.flatten())
    
    m = r[0] # Multiplier = slope
    b = r[1] # Offset = intercept

    arr1_norm = arr1 * m + b

    return (arr1_norm, (m,b))
      
def get_pos_ring(et, num_pts=100, radius = 122000, name_body='Jupiter', units='radec', wcs=False, 
                    frame='J2000', abcorr='LT+S', name_observer='New Horizons'):
    
    """
    Get an array of points for a ring, at a specified radius, seen from observer at the given ET.
    Result is in RA / Dec radians. If units='pixels', then it is in x y pixels, based on the supplied wcs.
    """
    
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

def imsize((size)):
    """
    Set plot size. Same as using rc, but easier syntax.
    """
    plt.rc('figure', figsize=(size[0], size[1]))
    
def correct_stellab(radec, vel):
    """
    Corect for stellar aberration.
    radec is array (n,2) in radians. velocity in km/sec. Both should be in J2K coords.
    """

    radec_abcorr = radec.copy()    
    for i in range(np.shape(radec)[0]):
        pos_i = cspice.radrec(1., radec[i,0], radec[i,1])
        pos_i_abcorr = cspice.stelab(pos_i, vel)
        rang, radec_abcorr[i,0], radec_abcorr[i,1] = cspice.recrad(pos_i_abcorr)

    return radec_abcorr
    
def image_from_list_points(points, shape, diam_kernel):
    """
    Given an ordered list of xy points, and an output size, creates an image.
    Useful for creating synthetic star fields.
    """
    
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

def dist_center(diam, circle=False, centered=True, invert=False, normalize=False):
    """
    Returns an array of dimensions diam x diam, with each cell being the distance from the center cell.
    Works best if diam is an odd integer.
    """
    
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

    return lcs_set.pop() # Original function returned lcs_set itself. I don't know why -- I just want to extract one element.

def sfit(arr, degree=3, binning=16): # For efficiency, we downsample the input array before doing the fit.
    """
    Fit polynomial to a 2D array, aka surface.
    """

# For info on resizing, see http://stackoverflow.com/questions/29958670/how-to-use-matlabs-imresize-in-python
    
    shape_small = (np.size(arr,0)/binning, np.size(arr,1)/binning)
    shape_big   = np.shape(arr)

# Create x and y arrays, which we need to pass to the fitting routine

    x_big, y_big = np.mgrid[:shape_big[0], :shape_big[1]]
    x_small = skt.resize(x_big, shape_small, order=1, preserve_range=True)
    y_small = skt.resize(y_big, shape_small, order=1, preserve_range=True)
    
    arr_small = skt.resize(arr, shape_small, order=1, preserve_range=True)
    p_init = astropy.modeling.models.Polynomial2D(degree=int(degree))

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

