#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Mon Nov 20 09:38:43 2017

@author: throop
"""

import math       
import matplotlib.pyplot as plt # pyplot
import numpy as np
from   astropy import units as u           # Units library
from   pymiecoated import Mie

def scatter_lambert(ang_phase, albedo=1):
    
    """
    Return the phase function of a Lambertian scatterer. Phase function is properly normalized.
    
    Parameters
    -----
    
    ang_phase:
        Phase angle. Radians, or astropy units. Must be in range 0 .. pi.
        
    albedo:
        Albedo. Optional.
        
    """
    
    pi = math.pi
    
    p11_lambert		 = 8/(3 * pi) * (np.sin(ang_phase) + (pi - ang_phase) * np.cos(ang_phase))
  
  			          # But hang on -- how is this phase function right since it doesn't include diffraction! Ugh...
  			          # The real phase function should really be half this, I think. This properly integrates to 
  			          # 2 over 0 .. 2 pi -- but what *should* integrate to 2 is the phase function including 
  			          # the diffraction spike, meaning these P11's in backscatter would go down.
  			          # OK, halving it. That will have the effect of making particles darker, harder to see, and 
  			          # increasing N for lambert.

                        # NB: At one point code was using "(pi-np.sin(ang_phase))" -- definite error.

  
    
    # Q: How do I write code so that it interacts nicely with both scalars and unit-ed arrays?
    # I can do    3.14 - 0    but not    3.13 - 0*u.rad
    
    p11_lambert      *= 0.5
  			          
    p11_lambert      *= albedo
    
#    p11_lambert_out       = np.clip(p11_lambert, 0, 1000) # None fails here. Numpy bug?
    
    return(p11_lambert)
    

    