#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 00:38:33 2020

@author: hthroop
"""

### 

# This reads a single Excel file provided by SARA, which lists the selection stats for 
# every single SMD program, in every single year. It then summarizes the results of these.
#
# The input file is collected manually by Max. It is not auto-generated from NSPIRES.
# I got it from https://science.nasa.gov/researchers/sara/grant-stats
#
# These results can be used to plot (for instance) the overall changes in selection rate vs. time,
# across the different SMD divisions: Planetary vs. Astro, etc. 
#
# I used these plots to give data to Julie @ JPL for her 2020 Decadal R&A White Paper.
#
#
# HBT 15-Sep-2020

import numpy as np
import xlrd as xlrd

import matplotlib.pyplot as plt
import numpy as np
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

import os.path
import pickle

import random
import csv
from scipy import stats   # For lin regress

from astropy.table import Table

from scipy.stats.stats import pearsonr

import hbt_short as hbt

import re
import glob

#%%

# First read in the Excel data

path_base = '/Users/hthroop/Documents/HQ/ProgramStats'
file = path_base + '/SelectionStatsSMD_Sep20.xls'

"""Read an excel file from NSPIRES"""

workbook = xlrd.open_workbook(os.path.join(path_base, file))

sheet_names = workbook.sheet_names()    

sheet = workbook.sheet_by_index(0)
num_rows = sheet.nrows
num_cols = sheet.ncols

name_columns = np.array(sheet.row_values(0))

data = {}  # Dictionary
# col_doctype = np.where(name_columns == 'Document Type')[0][0]

for i,name_column in enumerate(name_columns):
    vals = sheet.col_values(i)
    vals = vals[1:] # Remove the header row
    data[name_column] = np.array(vals)

# num_rows = len(data['ROSES Year'])

num_rows = 1239 # XXX fix this

col_year = np.where(name_columns == 'ROSES year')[0][0]
col_submitted = np.where(name_columns == 'Submitted ')[0][0]
col_selected = np.where(name_columns == 'Selected*')[0][0]
col_division = np.where(name_columns == 'SMD Division')[0][0]
col_program = np.where(name_columns == 'Solicitation or Program Element Title')[0][0]


year = []
program = []
n_submit = []
n_select = []
division = []

for i in range(num_rows):
    vals = sheet.row_values(i)

    year_i    = vals[col_year]
    program_i = vals[col_program]
    n_submit_i  = vals[col_submitted]
    n_select_i  = vals[col_selected]
    division_i  = vals[col_division]
    
    division_i = division_i.replace('division', 'Division') # Do a one-off correction. But it does not work properly.
    
    # Now check this row. Keep it only if it is good. We remove Step-1's and several other one-offs.
    
    if ( (hbt.is_number(year_i)) and (len(program_i) > 0) and 
         (hbt.is_number(n_submit_i)) and (hbt.is_number(n_select_i)) and
         ('Step-1' not in program_i) and
         ('Chandra Guest Investigator' not in program_i) and
         ('NOI' not in program_i) and
         ('Hubble Guest Observer' not in program_i)):
        
        # Now remove 'Step-2'. We have old programs ('CDAP') and new ones ('CDAP Step-2') and we want them merged.
        
        program_i = program_i.replace(' Step-2', '')
        
        # Change some one-off typos in the program names
        
        if (program_i == "Cassini Data Analysis: PDS Cassini Data Release 54"):
            program_i = "Cassini Data Analysis"

        if (program_i == "New Frontiers Data Analysis"):
            program_i = "New Frontiers Data Analysis Program"
            
        # program_i = program_i.repalce('')
        
        year.append(year_i)
        program.append(program_i)
        n_submit.append(n_submit_i)
        n_select.append(n_select_i)
        division.append(division_i)
        print(f'Adding program: {vals}')
    
    print(vals[0])

year = np.array(year).astype(int)
program = np.array(program)
n_submit = np.array(n_submit).astype(int)
n_select = np.array(n_select).astype(int)
division = np.array(division)

#%%

# Now make some plots!

# Plot # of programs per divsion, as a function of time.


hbt.figsize(12,9)
hbt.fontsize(10)

ms=4

programs_per_year = {} # Set up a dictionary

year_u = np.unique(year)
division_u = np.unique(division)

for d in division_u:
    programs_per_year[d] = []  # Blank list
    for y in year_u:
        n = np.sum(np.logical_and( (year == y), (division == d)))
        print(f'Division={d}, year={y}, n_programs = {n}')
        programs_per_year[d].append(n)

plt.subplot(2,2,1)

for d in division_u:
    plt.plot(year_u, programs_per_year[d], label=d, marker ='.', ms=ms)
plt.legend()
plt.xlabel('ROSES Year')
plt.ylabel('# Programs')   
plt.ylim([0,50]) 
# plt.show()
    
# plt.subplot(2,2,1)

submits_per_year = {} # Set up a dictionary
selects_per_year = {}

year_u     = np.unique(year)      # Get a list of individual years (i.e., 10, not 1200)
division_u = np.unique(division)

for d in division_u:
    submits_per_year[d] = []  # Blank list
    selects_per_year[d] = []  # Blank list
    for y in year_u:
        n = np.sum(n_submit * np.logical_and( (year == y), (division == d)))
        print(f'Division={d}, year={y}, n_submit = {n}')
        submits_per_year[d].append(n)

        n = np.sum(n_select * np.logical_and( (year == y), (division == d)))
        print(f'Division={d}, year={y}, n_select= {n}')
        selects_per_year[d].append(n)


plt.subplot(2,2,2)
for d in division_u:
    plt.plot(year_u, submits_per_year[d], label=d, marker ='.', ms=ms)
plt.xlabel('ROSES Year')
plt.ylabel('# Submits')    
plt.ylim([0, 2600])
plt.legend()
# plt.show()

plt.subplot(2,2,3)

for d in division_u:
    plt.plot(year_u, selects_per_year[d], label=d, marker ='.', ms=ms)
plt.xlabel('ROSES Year')
plt.ylabel('# Selects')    
plt.ylim([0, 900])
plt.legend()
# plt.show()

plt.subplot(2,2,4)

for d in division_u:
    plt.plot(year_u, np.array(selects_per_year[d]) / np.array(submits_per_year[d]), label=d, 
             marker ='.', ms=ms)
plt.legend()
plt.xlabel('ROSES Year')
plt.ylim([0.1, 0.5])     # Hand-tune this to get the legend in the right place
plt.ylabel('% Select')
plt.show()

#%%

hbt.fontsize(20)

for d in ['Planetary Science', 'Astrophysics']:
    plt.plot(year_u, np.array(selects_per_year[d]) / np.array(submits_per_year[d]), label=d, 
             marker ='.', ms=ms)
plt.legend()
plt.xlabel('ROSES Year')
plt.ylim([0.1, 0.5])     # Hand-tune this to get the legend in the right place
plt.ylabel('% Select')
plt.show()


#%% list all the programs in a year. This is more for debugging.

d = ['Planetary Science']
for d_i in d:
    for y_i in year_u:
        w = np.where(np.logical_and( (division == d_i), (year==y_i)))[0]
        print(f'{d_i}, {y_i}, N = {len(w)} programs')
        for w_i in w:
            print(f'  {year[w_i]}, n={n_select[w_i]} / {n_submit[w_i]}, {program[w_i]}')
        print()    
    
#%% Now make a list, program-by-program. This will list which years each program is active.

d = ['Planetary Science', 'Astrophysics']
d = ['Cross Division']

for d_i in d:

    for p_i in np.unique(program):  # Loop through all of the program names
        
        w = np.where(np.logical_and( (division == d_i), (program == p_i)))[0]
        if len(w) > 0:
            # print (f'{p_i}')
            n_select_i = 0
            n_submit_i = 0
            for w_i in w[::-1]:
                y0 = year[w[-1]]
                y1 = year[w[0]]
                n_select_i += n_select[w_i]
                n_submit_i += n_submit[w_i]
                
                # print(f'  {year[w_i]}, n={n_select[w_i]} / {n_submit[w_i]}, {program[w_i]}')
            print(f' N={len(w):2} years, {y0}-{y1}; n = {n_select_i:4} / {n_submit_i:4} proposals; {program[w_i]}')    
            # print()    
            

        
