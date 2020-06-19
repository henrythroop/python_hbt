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
import astropy

def scatter_isotropic(ang_phase_in, albedo=1):
    
    """
    Return the phase function of an isotropic scatterer. Phase function is properly normalized, I hope, but not sure.
    
    Parameters
    -----
    
    ang_phase:
        Phase angle. Radians, or astropy units. Must be in range 0 .. pi.
        Since it is isotropic, this value is ignored, duh.
        
    albedo:
        Albedo. Optional.
        
    """    
        
    p11_isotropic		 = albedo + np.zeros(len(ang_phase_in))
    
    return(p11_isotropic)
    

    