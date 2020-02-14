#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 22:22:10 2020

@author: throop
"""

import numpy as np
import xlrd as xlrd

import hbt
import re

def abbreviate(s):

    a = [
    ('Johns Hopkins University/Applied Physics Laboratory', 'JHU-APL'),
    ('SETI Institute', 'SETI'),
    ('Jet Propulsion Laboratory', 'JPL'),
    ('NASA Ames Research Center', 'NASA Ames'),
    ('Lowell Observatory', 'Lowell'),
    ('Johns Hopkins University', 'JHU'),
    ('University of Maryland, College Park', 'UMD'),
    ('Brown University', 'Brown U'),
    ('NASA Johnson Space Center', 'NASA JSC'),
    ('University Of Colorado, Boulder', 'U Colorado'),
    ('Southwest Research Institute', 'SwRI'),
    ('Arizona State University', 'ASU'),
    ('University of New Mexico', 'UNM'),
    ('University Of California, Santa Cruz', 'UCSC'),
    ('NASA Foreign PI Support Organization', 'Foreign'),
    ('Florida Institute Of Technology', 'FIT'),
    ('State University Of New York, ', 'SUNY '),
    ('Lawrence Livermore National Security, LLC', 'LLNL'),
    ('University Of Idaho, Moscow', 'U Idaho'),
    ('University Of Arizona', 'U Arizona'),
    ('University of California, Los Angeles', 'UCLA'),
    ('University of Hawaii, Honolulu', 'U Hawaii'),
    ('Purdue University', 'Purdue'),
    ('University of Western Ontario', 'U Western Ontario'),
    ('Planetary Science Institute', 'PSI'),
    ('Auburn University', 'Auburn'),
    ('State University', 'State'),
    ('Corporation', ''),
    
    (', LLC', ''),
    ('Imperial College Of Science Technology & Medicine', 'Imperial College')]

    for t in a:
        insensitive = re.compile(t[0], re.IGNORECASE)
        s = insensitive.sub(t[1], s)
        # s = s.replace(t[0], t[1])
        
    return s

file_xl = '/Users/throop/HQ/python/Panel_Compilation_ImpactProcesses.xls'

workbook = xlrd.open_workbook(file_xl)
sheet_names = workbook.sheet_names()
print('Sheet Names', sheet_names)

num_proposals = len(sheet_names)-3
sheet_proposals = workbook.sheet_propoals

for i in range(num_proposals):
    sheet = workbook.sheet_by_index(i+3)
    num_investigators = sheet.nrows-1

    for j in range(num_investigators):
        
        role = sheet.cell_value(j+1,0)  # row (1st row is 0), column (1st col is A)
        role = role.replace('Collaborator', 'Collab').replace('Science PI', 'Sci-PI')
        role = role.replace('Graduate/Undergraduate Student', 'Student')
        # Add another line for Postdoctoral Scholar
        
        name_last = sheet.cell_value(j+1,3)
        name_first = sheet.cell_value(j+1,4)
        if (role == 'PI'):
            institution = sheet.cell_value(j+1,7)
        else:
            institution = sheet.cell_value(j+1,6)
        
        institution_short = abbreviate(institution)
        # print(f'Shortened {institution} to {institution_short}')
        
        # XXX Add a line to print the proposal number and title here.
        
        line = (f'{role} {name_first} {name_last} / {institution_short}')
        print(line)
    print()