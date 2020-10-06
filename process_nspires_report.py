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

path_base = '/Users/hthroop/Documents/HQ/ProgramStats'
# files_xl = glob.glob(path_base + '/SSW*-out.xls')
# files_xl = glob.glob(path_base + '/HW*-out.xls')
# files_xl = glob.glob(path_base + '/CDAP*-out.xls')
# files_xl = glob.glob(path_base + '/NFDAP*-out.xls')
# files_xl = glob.glob(path_base + '/EW*-out.xls')
files_xl = glob.glob(path_base + '/PDART*-out.xls')

# file_xl = 'SSW19-out.xls'
files_xl = np.sort(files_xl)[::1]
ii = 1

def read_excel_nspires(file):

    """Read an excel file from NSPIRES"""

    workbook = xlrd.open_workbook(os.path.join(path_base, file))
    
    sheet_names = workbook.sheet_names()    
    
    sheet = workbook.sheet_by_index(0)
    num_rows = sheet.nrows
    num_cols = sheet.ncols
    
    name_columns = np.array(sheet.row_values(0))
    
    data = {}  # Dictionary
    col_doctype = np.where(name_columns == 'Document Type')[0][0]
    
    for i,name_column in enumerate(name_columns):
        vals = sheet.col_values(i)
        vals = vals[1:] # Remove the header row
        if ('udget' in name_column) or ('Proposed Amount Total' in name_column):
            data[name_column] = np.array(vals).astype(int)  # Convert to int, if possible
        else:    
            data[name_column] = np.array(vals)

    return (data, np.array(list(data.keys())))

##########
# Start main code
########## 

# Damn! CDAP06 data breaks this. Its columns are not aligned with the data properly. Corruption on 
# NSPIRES side.    
           
hbt.fontsize(18)

hbt.figsize(20,15)

year = []
n_submit = []
n_select = []
budget_total_select = []
budget_total_submit = []

for file in files_xl:
 
    (data, name_columns) = read_excel_nspires(file)
    
    t = Table(data)  # For fun, make an Astropy table. Not used yet.
    
    plt.subplot(4,4,ii)
    
    # We seem to have an entry for every selection document (tech eval, letter, etc).
    # Remove most of these, leaving just one per proposal.
        
    where_letter = np.where(data['Document Type'] == 'Notification Letter')
    where_letter = np.where(data['Document Type'] == 'Panel Evaluation') # CDAP13 does *not* have a notification letter for most.
    
    for name_column in name_columns:
        data[name_column] = data[name_column][where_letter]
    
    name_program  = (data['Proposal Number'][0].split('-')[1]).replace('_2', '')
    bins = np.arange(0,1300000,50000)/1000
    where_selected = np.where(np.logical_or( data['Selection Status'] == 'Selected',
                                             data['Selection Status'] == 'Selected (Partial)' ))
    
    num_proposals = len(data[name_column])
    num_selected  = len(data[name_column][where_selected])
    
    mean_y1       = np.nanmean(data['Proposed Budget Amount(by Year) 1'])
    median_y1     = np.nanmedian(data['Proposed Budget Amount(by Year) 1'])
    tot_y1        = np.nansum(data['Proposed Budget Amount(by Year) 1'])
    mean_y1_sel   = np.nanmean(data['Proposed Budget Amount(by Year) 1'][where_selected])
    median_y1_sel = np.nanmedian(data['Proposed Budget Amount(by Year) 1'][where_selected])
    tot_all       = np.nansum(data['Proposed Amount Total'])
    tot_all_sel   = np.nansum(data['Proposed Amount Total'][where_selected])
    
    plt.hist(data['Proposed Amount Total']/1000, bins,
             alpha = 0.5,
             label = f'All, mean=${tot_all/1000/num_proposals:.0f}K')
    plt.hist(data['Proposed Amount Total'][where_selected]/1000, bins,
             alpha = 0.5, color='green',
             label = f'Selected, mean=${tot_all_sel/1000/num_selected:.0f}K')
    plt.legend()
    plt.title(f'{name_program}, N = {num_selected}/{num_proposals} = {100*num_selected/num_proposals:.0f}%')
    plt.xlabel('Proposed Budget [$K]')
    plt.ylabel('Number')
    plt.tight_layout()
    # plt.show()
    
    print(f'{name_program}')
    print(f' Y1            Mean, Median: ${mean_y1/1000:.0f}K, ${median_y1/1000:.0f}K')
    print(f' Y1 Selected   Mean, Median: ${mean_y1_sel/1000:.0f}K, ${median_y1_sel/1000:.0f}K')
    print(f' Y1            Total:        ${tot_y1/1e6:.0f}M')
    print(f' Y1-3          Total:        ${tot_all/1e6:.0f}M')
    print(f' Y1-3 Selected Total:        ${tot_all_sel/1e6:.0f}M')
    print(f' Selected           :        {num_selected} / {num_proposals} = {100*num_selected/num_proposals:5.1f}%')
    

    year.append(name_program.replace('/2','')[-2::])  # Convert SSW19 and CDAPS12/2 into 19 and 12
    n_submit.append(num_proposals)
    n_select.append(num_selected)
    budget_total_select.append(tot_all_sel)
    budget_total_submit.append(tot_all)
    
    print()
    ii+=1

plt.show()

name_program_short = name_program[0:-2]

year = np.array(year).astype(int)
n_submit = np.array(n_submit)
n_select = np.array(n_select)
budget_total_select = np.array(budget_total_select)
budget_total_submit = np.array(budget_total_submit)

#%%%

hbt.fontsize(20)

plt.subplot(2,2,1)
plt.plot(year, n_submit, label = '# Submit', ms=20, marker='.')
plt.plot(year, n_select, label = '# Select', ms=20, marker='.')
plt.legend()
plt.ylabel('# Proposals')
plt.xlabel('Year')
plt.title(name_program_short)

plt.subplot(2,2,3)
plt.plot(year, budget_total_submit/1e6, label = '$M Submit', ms=20, marker='.')
plt.plot(year, budget_total_select/1e6, label = '$M Select', ms=20, marker='.')
plt.legend()
plt.ylabel('$M Total')
plt.xlabel('Year')
plt.title(name_program_short)

plt.subplot(2,2,2)
plt.plot(year, budget_total_submit/n_submit/1e3, label = 'Mean Budget, Submitted', ms=20, marker='.')
plt.plot(year, budget_total_select/n_select/1e3, label = 'Mean Budget, Selected', ms=20, marker='.')
plt.legend()
plt.ylabel('$K Per Proposal')
plt.xlabel('Year')
plt.title(name_program_short)

plt.subplot(2,2,4)
plt.plot(year, 100* n_select / n_submit, label = '% Select', ms=20, marker = '.')
plt.legend()
plt.ylabel('% Select')
plt.xlabel('Year')
plt.title(name_program_short)

plt.show()

