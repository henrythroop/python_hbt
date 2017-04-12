#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 20:53:33 2017

@author: throop
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import astropy
from   astropy.table import Table
import astropy.table as table

import astropy.units as u
import astropy.constants as c
import os.path # For expanduser

from astropy.io import ascii

import glob

files_in = glob.glob('/Users/throop/Desktop/OmniFocus*csv')

for file in files_in:
    if '_fixed' not in file:
        print("Reading: " + file)
        table = ascii.read(file, format='ecsv')
        status_fixed = np.zeros(np.shape(table)[0], dtype=u'S20')
        completed = table['Completion Date'].mask
        for i in range(np.shape(table)[0]):
            status_fixed[i] = 'Completed' if completed[i] else ''
        table['StatusFixed'] = status_fixed
    
#    del table['Status']
#    del table['Task ID']
#    del table['Due Date']
        
        file_out = file.replace('.csv', '_fixed.csv')      
        ascii.write(table, file_out, format = 'csv')
        print("Wrote: " + file_out)
    
