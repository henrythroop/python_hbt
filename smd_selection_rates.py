#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 00:38:33 2020

@author: hthroop
"""

### 

# This reads output from I-NSPIRES and makes plots.

# I have used it to get the total $$ requested for SSW, as a function of time.
# This code does not look at scoresheets. It only looks at the NSPIRES output.

# To create the output from I-NSPIRES:
# - Output as .xls.
# - Include all columns. Better to have too many, than too few. Code will find the proper ones. 
# - Be aware that there is a bug in NSPIRES that causes some columns to be mis-labeled (e.g., CDAP06).

# HBT 1-Jul-2020

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

num_rows = 1145 # XXX fix this

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
    
    # Now check this row. Keep it only if it is good.
    
    if ( (hbt.is_number(year_i)) and (len(program_i) > 0) and 
         (hbt.is_number(n_submit_i)) and (hbt.is_number(n_select_i)) ):
        
        year.append(year_i)
        program.append(program_i)
        n_submit.append(n_submit_i)
        n_select.append(n_select_i)
        division.append(division_i)
    
    print(vals[0])

year = np.array(year)
program = np.array(program)
n_submit = np.array(n_submit)
n_select = np.array(n_select)
division = np.array(division)

#%%

# Now make some plots!

programs_per_year = {} # Set up a dictionary

year_u = np.unique(year)
division_u = np.unique(division)

for d in division_u:
    programs_per_year[d] = []  # Blank list
    for y in year_u:
        n = np.sum(np.logical_and( (year == y), (division == d)))
        print(f'Division={d}, year={y}, n_programs = {n}')
        programs_per_year[d].append(n)

for d in division_u:
    plt.plot(year_u, programs_per_year[d], label=d)
plt.legend()
plt.xlabel('ROSES Year')
plt.ylabel('# Programs')    
plt.show()
    
#%%


submits_per_year = {} # Set up a dictionary
selects_per_year = {}

year_u = np.unique(year)
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

for d in division_u:
    plt.plot(year_u, selects_per_year[d], label=d)
plt.legend()
plt.xlabel('ROSES Year')
plt.ylabel('# Selects')    
plt.show()

for d in division_u:
    plt.plot(year_u, submits_per_year[d], label=d)
plt.legend()
plt.xlabel('ROSES Year')
plt.ylabel('# Submits')    
plt.show()

for d in division_u:
    plt.plot(year_u, np.array(selects_per_year[d]) / np.array(submits_per_year[d]), label=d)
plt.legend()
plt.xlabel('ROSES Year')
plt.ylabel('% Select')    
plt.show()

