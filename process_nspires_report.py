#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 00:38:33 2020

@author: hthroop
"""

### 

# This reads output from I-NSPIRES and makes plots.

# Will make plots such as:
#   - Total $$ requested for SSS vs. time.
#   - Selection rate of SSW, HW, and PDART vs. time
#   - Average proposal size of all programs vs. time
#
# This code looks at the output from NSPIRES, as saved into .xls files.
# It does not look at Max's SMD data, and it does not look at individual score sheets.
#
# One limitation in the output: it does look at requested proposal sizes. But, if a proposal is descoped,
# that will usually not be noticed here. (Sometimes yes, sometimes no -- depending on if NRESS put the award
# sizes into NSPIRES or not.)

# To create the output from I-NSPIRES:
# - Selection Module → View All → Select All → Export Proposal(s).
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
import sys

def process_nspires_report():
"""
    This is the wrapper program that runs the code. Each program (e.g., CDAP) is run individually,
    and then the data are plotted together.
    
    This code can be cut + pasted to command line easily.

    Returns
    -------
    None.

    """

#%%%    
    programs = ['SSW', 'PDART', 'SSO', 'HW', 'LDAP', 'MDAP', 'CDAP', 'NFDAP']
    
    # NB: All programs work, but SSO has some strangeness (e.g., 2015, 2016, 2018 mean proposal sizes are > $1M/yr)
    #
    # programs = ['SSW', 'HW', 'PDART']
    # programs = ['LDAP','MDAP']
    # programs = ['SSW']
    # programs = ['MDAP']
    # path_program = 'CDAP/NSPIRES/'
    # path_program = 'HW/NSPIRES/'
    # path_program = 'SSO/NSPIRES/'

    out = {}
    
    
    for program in programs:
        out[program] = read_nspires_multiyear(program)

# Make some plots
        
    hbt.fontsize(30)

    for program in programs:
        a = out[program]
        plt.plot( a['year']+2000, a['frac_select']*100, label=program, marker='.', ms=25, linewidth=3)
    plt.legend()
    plt.xlabel('ROSES Year')
    plt.ylabel('Selection Rate by # Proposals [%]')
    plt.ylim([0,50])
    plt.show()

    for program in programs:
        a = out[program]
        plt.plot( a['year']+2000, a['frac_budget_total_select']*100, label=program, marker='.', ms=25, linewidth=3)
    plt.legend()
    plt.xlabel('ROSES Year')
    plt.ylabel('Selection Rate by $ Requested [%]')
    plt.ylim([0,50])
    plt.show()

    for program in programs:
        a = out[program]
        plt.plot( a['year']+2000, a['budget_total_submit'] / a['n_submit']/1000, label=program, marker='.', ms=25, linewidth=3)
    plt.legend()
    plt.xlabel('ROSES Year')
    plt.ylabel('Mean Proposal Size [$K]')
    # plt.ylim([0,50])
    plt.show()
    
#%%% 
    f  
    
def read_excel_nspires(file):

    """Read an excel file from NSPIRES"""

    workbook = xlrd.open_workbook(file)
    
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

def read_nspires_multiyear(name_program_in):
    
    path_base = '/Users/hthroop/Documents/HQ/ProgramStats/'
    
    path_program = os.path.join(name_program_in, 'NSPIRES')
    
    # path_program = 'PDART/NSPIRES/'
    # path_program = 'SSW/NSPIRES/'
    # path_program = 'NFDAP/NSPIRES/'
    # path_program = 'EW/NSPIRES/'
     
    # files_xl = glob.glob(path_base + '/SSW*-out.xls')
    # files_xl = glob.glob(path_base + '/HW*-out.xls')
    # files_xl = glob.glob(path_base + '/CDAP/NSPIfRES/*-out.xls')
    # files_xl = glob.glob(path_base + '/NFDAP/NSPIRES/*-out.xls')
    # files_xl = glob.glob(path_base + '/EW*-out.xls')
    # files_xl = glob.glob(path_base + '/PDART*-out.xls')
    
    # file_xl = 'SSW19-out.xls'
    
    files_xl = glob.glob(os.path.join(path_base, path_program, '*-out.xls'))
    
    files_xl = np.sort(files_xl)[::1]
    ii = 1
    
    if len(files_xl) == 0:
        sys.exit(f'0 files found in path {path_base + path_program}')

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
    
    # Loop over the files, and process each one, year by year
    
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
    
    frac_select = n_select / n_submit
    frac_budget_total_select = budget_total_select / budget_total_submit
    
    
    # return(year, n_submit, n_select, frac_select, budget_total_submit, budget_total_select, frac_budget_total_select))

    return({'year':year, 'n_submit':n_submit, 'n_select':n_select, 'frac_select':frac_select, 
           'budget_total_submit':budget_total_submit, 'budget_total_select':budget_total_select, 
           'frac_budget_total_select':frac_budget_total_select})

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

#%%%
# Print a data table for easy export to Excel etc.

    print(f'{name_program_short}')
    print('year, n_submit, n_select, budget_total_submit, budget_total_select' )
    for i,year_i in enumerate(year):
        print(f'{year[i]+2000}, ' + \
              f'{n_submit[i]}, {n_select[i]}, {n_select[i] / n_submit[i]}, ' + \
              f'{budget_total_submit[i]}, {budget_total_select[i]}, {budget_total_select[i] / budget_total_submit[i]} ')
    

# =============================================================================
# Call the main code
# =============================================================================

if (__name__ == '__main__'):
    process_nspires_report()



    
