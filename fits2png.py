#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  1 15:27:05 2023

@author: throop
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 22:56:57 2023

@author: throop
"""

# from hbt_short import hbt 

# import hbt_short

import glob
import os.path
import os
import subprocess
import datetime
import matplotlib.pyplot as plt
import math
import numpy as np
import astropy
import astropy.modeling
#import matplotlib.pyplot as plt # Define in each function individually
import matplotlib                # We need this to access 'rc'
# import spiceypy as sp
from   astropy.io import fits
import subprocess
# import hbt
import warnings
import importlib  # So I can do importlib.reload(module)
# from   photutils import DAOStarFinder
# import photutils
from   astropy import units as u           # Units library
import scipy.ndimage
from astropy.time import Time
from astropy.visualization import PercentileInterval

from   scipy.optimize import curve_fit
from   astropy.convolution import Box1DKernel, Gaussian1DKernel, convolve
from PIL import Image
from scipy import ndimage
from astropy.visualization import simple_norm
from astropy.modeling import models
from astropy.modeling.models import Polynomial1D
from astropy.modeling.fitting import LinearLSQFitter
from astropy.modeling import fitting


import scipy.misc
import pytz

from suncalc import get_position, get_times
from datetime import datetime, timezone, timedelta

def fits2png():

    # This is just a purely dumb FITS → PNG converter.
    # No rotating, flattening, etc.
    # Useful for solar eclipse, where the centroiding was (of course) not working properly.
    
    # HBT 1-May-2023
    
    DO_CROP = False
    
    path_in = '/Volumes/Data/Solar/Eclipse_20Apr23'
    path_out = os.path.join(path_in, 'out')
    
    if not(os.path.exists(path_out)):
        os.mkdir(path_out)
        print('Create: ' + path_out)
        
    files = glob.glob(os.path.join(path_in, '*.fit*'))
    files = np.sort(files)
        
    plt.set_cmap(plt.get_cmap('Greys_r'))

    for i,file in enumerate(files):
    
        if DO_CROP:
            img_c = img[500:3000, 1000:3500] 
        else:
            img_c = img
        

        plt.imshow(img_c)
        plt.show()
    
        file_out = file.replace(path_in, path_out).replace( '.fit', '.png')
        img_pil   = Image.fromarray(img_c)
        img_pil.save(file_out)

        print(f'{i}/{len(files)}: Wrote: ' + file_out)


if (__name__ == '__main__'):
    fits2png()        

