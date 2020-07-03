#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 00:38:33 2020

@author: hthroop
"""

### This reads output from I-NSPIRES and makes plots.
# Output as .xls.
# Include all columns. Better to have too many, than too few.
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

path_base = '/Users/hthroop/Downloads'
files_xl = glob.glob(path_base + '/SSW*-out.xls')
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

for file in files_xl:
 
    (data, name_columns) = read_excel_nspires(file)
    
    t = Table(data)  # For fun, make an Astropy table. Not used yet.
    
    plt.subplot(4,4,ii)
    
    # We seem to have an entry for every selection document (tech eval, letter, etc).
    # Remove most of these, leaving just one per proposal.
        
    where_letter = np.where(data['Document Type'] == 'Notification Letter')
    
    for name_column in name_columns:
        data[name_column] = data[name_column][where_letter]
    
    name_program  = (data['Proposal Number'][0].split('-')[1]).replace('_2', '')
    bins = np.arange(0,500000,25000)/1000
    where_selected = np.where(data['Selection Status'] == 'Selected')
    
    num_proposals = len(data[name_column])
    num_selected  = len(data[name_column][where_selected])
    
    mean_y1 = np.nanmean(data['Proposed Budget Amount(by Year) 1'])
    median_y1 = np.nanmedian(data['Proposed Budget Amount(by Year) 1'])
    mean_y1_sel = np.nanmean(data['Proposed Budget Amount(by Year) 1'][where_selected])
    median_y1_sel = np.nanmedian(data['Proposed Budget Amount(by Year) 1'][where_selected])
    tot_all = np.nansum(data['Proposed Amount Total'])
    tot_all_sel = np.nansum(data['Proposed Amount Total'][where_selected])
    
    plt.hist(data['Proposed Budget Amount(by Year) 1']/1000, bins,
             alpha = 0.5,
             label = f'All, mean=${mean_y1/1000:.0f}K')
    plt.hist(data['Proposed Budget Amount(by Year) 1'][where_selected]/1000, bins,
             alpha = 0.5, color='green',
             label = f'Selected, mean=${mean_y1_sel/1000:.0f}K')
    plt.legend()
    plt.title(f'{name_program}, N = {num_selected}/{num_proposals} = {100*num_selected/num_proposals:.0f}%')
    plt.xlabel('Proposed Y1 Budget [$K]')
    plt.ylabel('Number')
    plt.tight_layout()
    # plt.show()
    
    print(f'{name_program}')
    print(f' Y1            Mean, Median: ${mean_y1/1000:.0f}K, ${median_y1/1000:.0f}K')
    print(f' Y1 Selected   Mean, Median: ${mean_y1_sel/1000:.0f}K, ${median_y1_sel/1000:.0f}K')
    print(f' Y1-3          Total:        ${tot_all/1e6:.0f}M')
    print(f' Y1-3 Selected Total:        ${tot_all_sel/1e6:.0f}M')
    
    print()
    ii+=1

plt.show()
    

